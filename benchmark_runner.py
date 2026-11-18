#!/usr/bin/env python3
"""Headless benchmark runner for Topic 2 (prompt keyword effects on LLM decode efficiency).

What it does
  * Finds every .gguf in ./models (smallest first) and loads prompts/*.json.
  * Runs EACH MODEL IN ITS OWN SUBPROCESS so ru_maxrss (a lifetime peak) is clean per model
    and memory is fully released between models.
  * Reads exact llama.cpp C-level timers (t_p_eval_ms, t_eval_ms, n_p_eval, n_eval) through
    llama_cpp.llama_perf_context(); falls back to wall-clock streaming timers if unavailable.
  * Records peak RSS (MB) from resource.getrusage(RUSAGE_SELF).ru_maxrss (bytes on macOS).
  * Streams one row per run into benchmark_results.csv. Re-running resumes: finished
    (model, prompt_id, trial) rows are skipped.

Typical use
  python benchmark_runner.py --dry-run         # counts + real token lengths, no benchmarking
  python benchmark_runner.py                   # full run
  python benchmark_runner.py --only tinyllama  # one model (substring of the file name)
"""
import argparse
import csv
import gc
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import resource
except ImportError:  # not expected on macOS
    resource = None

CSV_FIELDS = [
    "timestamp", "model", "prompt_id", "category", "topic", "length", "keyword", "trial",
    "cold_start", "timing_source",
    "raw_prompt_tokens", "input_tokens", "output_tokens", "total_tokens",
    "t_prefill_ms", "t_decode_ms", "decode_tokens", "decode_tps", "prefill_tps",
    "ms_per_output_token", "ttft_wall_ms", "wall_ms",
    "peak_rss_mb", "kv_cache_est_mb",
    "n_threads", "n_ctx", "temperature", "max_tokens", "finish_reason",
]
PROMPT_ORDER = ["factual", "reasoning", "role_based", "instruction", "creative"]
MODEL_LABELS = [("tinyllama", "TinyLlama-1.1B"), ("qwen", "Qwen2.5-1.5B"), ("phi", "Phi-3-mini")]


# ----------------------------------------------------------------------------- helpers
def physical_cores() -> int:
    try:
        return int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"]).decode().strip())
    except Exception:
        return max(1, (os.cpu_count() or 2) // 2)


def model_label(path: Path) -> str:
    name = path.name.lower()
    for key, label in MODEL_LABELS:
        if key in name:
            return label
    return path.stem


def find_models(models_dir, only):
    files = [p for p in Path(models_dir).rglob("*.gguf")]
    if only:
        files = [p for p in files if any(o.lower() in p.name.lower() for o in only)]
    return sorted(files, key=lambda p: p.stat().st_size)


def load_prompts(prompts_dir):
    items = []
    for name in PROMPT_ORDER:
        p = Path(prompts_dir) / f"{name}.json"
        if p.exists():
            items += json.loads(p.read_text())
    for p in sorted(Path(prompts_dir).glob("*.json")):
        if p.stem not in PROMPT_ORDER:
            items += json.loads(p.read_text())
    return items


def ensure_header(path):
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def done_keys(path):
    keys = set()
    if Path(path).exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    keys.add((r["model"], r["prompt_id"], int(r["trial"])))
                except (KeyError, ValueError):
                    pass
    return keys


def peak_rss_mb() -> float:
    if resource is None:
        return float("nan")
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports BYTES, Linux reports KILOBYTES.
    return r / 1024 / 1024 if sys.platform == "darwin" else r / 1024


def get_ctx(llm):
    for getter in (lambda: llm._ctx.ctx, lambda: llm.ctx):
        try:
            c = getter()
            if c is not None:
                return c
        except Exception:
            continue
    return None


def kv_estimate_mb(llm, n_tokens):
    """KV cache size estimate (f16 K and V): 2 * layers * kv_heads * head_dim * 2 bytes * tokens."""
    try:
        md = llm.metadata
        arch = md["general.architecture"]
        layers = int(md[f"{arch}.block_count"])
        heads = int(md[f"{arch}.attention.head_count"])
        kv_heads = int(md.get(f"{arch}.attention.head_count_kv", heads))
        emb = int(md[f"{arch}.embedding_length"])
        return 2 * layers * kv_heads * (emb // heads) * 2 * n_tokens / 1024 / 1024
    except Exception:
        return float("nan")


# ----------------------------------------------------------------------------- one run
def run_one(llm, llama_cpp, text, a):
    llm.reset()  # drop any cached prefix so every run pays a full prefill
    try:
        llm.set_seed(a.seed)
    except Exception:
        pass

    ctx = get_ctx(llm)
    perf_ok = False
    if ctx is not None and hasattr(llama_cpp, "llama_perf_context_reset"):
        try:
            llama_cpp.llama_perf_context_reset(ctx)
            perf_ok = True
        except Exception:
            perf_ok = False

    t0 = time.perf_counter()
    t_first = None
    n_chunks = 0
    pieces = []
    finish = ""
    stream = llm.create_chat_completion(
        messages=[{"role": "user", "content": text}],
        max_tokens=a.max_tokens, temperature=a.temperature, top_p=0.95, top_k=40, stream=True,
    )
    for chunk in stream:
        ch = chunk["choices"][0]
        piece = ch.get("delta", {}).get("content")
        if piece:
            if t_first is None:
                t_first = time.perf_counter()
            n_chunks += 1
            pieces.append(piece)
        if ch.get("finish_reason"):
            finish = ch["finish_reason"]
    t1 = time.perf_counter()

    wall_ms = (t1 - t0) * 1000
    ttft_ms = ((t_first or t1) - t0) * 1000

    source = "wall"
    t_p = ttft_ms
    t_e = max(wall_ms - ttft_ms, 0.0)
    n_p = 0
    n_e = max(n_chunks - 1, 0)
    if perf_ok:
        try:
            d = llama_cpp.llama_perf_context(ctx)
            if d.t_p_eval_ms > 0 and d.n_p_eval > 0:
                t_p, t_e, n_p, n_e = d.t_p_eval_ms, d.t_eval_ms, d.n_p_eval, d.n_eval
                source = "c_perf"
        except Exception:
            pass

    return {
        "timing_source": source,
        "input_tokens": n_p,
        "output_tokens": n_chunks,
        "t_prefill_ms": t_p,
        "t_decode_ms": t_e,
        "decode_tokens": n_e,
        "decode_tps": (n_e / (t_e / 1000)) if t_e > 0 and n_e > 0 else float("nan"),
        "prefill_tps": (n_p / (t_p / 1000)) if t_p > 0 and n_p > 0 else float("nan"),
        "ms_per_output_token": (t_e / n_e) if n_e > 0 else float("nan"),
        "ttft_wall_ms": ttft_ms,
        "wall_ms": wall_ms,
        "finish_reason": finish,
        "text": "".join(pieces),
    }


# ----------------------------------------------------------------------------- worker
def worker(a):
    import llama_cpp
    from llama_cpp import Llama

    model_path = Path(a.model)
    label = model_label(model_path)
    prompts = load_prompts(a.prompts_dir)
    done = done_keys(a.out)
    ensure_header(a.out)

    todo = [(t, p) for t in range(1, a.trials + 1) for p in prompts if (label, p["id"], t) not in done]
    if not todo:
        print(f"[{label}] nothing to do (all rows already in {a.out})")
        return
    size_gb = model_path.stat().st_size / 1e9
    print(f"[{label}] {len(todo)} runs to do, file {size_gb:.2f} GB, threads={a.threads}", flush=True)
    if size_gb > 2.6:
        print(f"[{label}] WARNING: file is over ~2.5 GB; watch Memory Pressure for swapping.", flush=True)

    llm = Llama(model_path=str(model_path), n_ctx=a.n_ctx, n_threads=a.threads,
                n_threads_batch=a.threads, n_batch=512, seed=a.seed, verbose=False)

    out_f = open(a.out, "a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=CSV_FIELDS)
    txt_f = open(a.outputs, "a")
    first = True
    try:
        for i, (trial, p) in enumerate(todo, 1):
            raw_tokens = len(llm.tokenize(p["prompt"].encode("utf-8"), add_bos=True))
            r = run_one(llm, llama_cpp, p["prompt"], a)
            n_in = r["input_tokens"] or raw_tokens
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": label, "prompt_id": p["id"], "category": p["category"],
                "topic": p.get("topic", ""), "length": p["length"], "keyword": p["keyword"],
                "trial": trial, "cold_start": int(first), "timing_source": r["timing_source"],
                "raw_prompt_tokens": raw_tokens, "input_tokens": n_in,
                "output_tokens": r["output_tokens"],
                "total_tokens": n_in + r["output_tokens"],
                "t_prefill_ms": round(r["t_prefill_ms"], 3), "t_decode_ms": round(r["t_decode_ms"], 3),
                "decode_tokens": r["decode_tokens"],
                "decode_tps": round(r["decode_tps"], 4), "prefill_tps": round(r["prefill_tps"], 4),
                "ms_per_output_token": round(r["ms_per_output_token"], 4),
                "ttft_wall_ms": round(r["ttft_wall_ms"], 3), "wall_ms": round(r["wall_ms"], 3),
                "peak_rss_mb": round(peak_rss_mb(), 2),
                "kv_cache_est_mb": round(kv_estimate_mb(llm, n_in + r["output_tokens"]), 2),
                "n_threads": a.threads, "n_ctx": a.n_ctx, "temperature": a.temperature,
                "max_tokens": a.max_tokens, "finish_reason": r["finish_reason"],
            }
            writer.writerow(row)
            out_f.flush()
            txt_f.write(json.dumps({"model": label, "prompt_id": p["id"], "trial": trial,
                                    "text": r["text"]}) + "\n")
            txt_f.flush()
            first = False
            print(f"[{label}] {i}/{len(todo)} {p['id']} t{trial}: in={n_in} out={r['output_tokens']} "
                  f"prefill={r['t_prefill_ms']:.0f}ms decode={r['decode_tps']:.1f} tok/s "
                  f"rss={row['peak_rss_mb']:.0f}MB [{r['timing_source']}]", flush=True)
    finally:
        out_f.close()
        txt_f.close()
        del llm
        gc.collect()


# ----------------------------------------------------------------------------- parent
def dry_run(models, prompts):
    from llama_cpp import Llama
    llm = Llama(model_path=str(models[0]), n_ctx=2048, verbose=False)
    by_len = {}
    for p in prompts:
        by_len.setdefault(p["length"], []).append(len(llm.tokenize(p["prompt"].encode(), add_bos=True)))
    print(f"\nReal token counts using {models[0].name}:")
    for length in ("short", "medium", "long"):
        v = sorted(by_len.get(length, []))
        if v:
            print(f"  {length:<7} n={len(v):<3} min={v[0]:<5} median={v[len(v)//2]:<5} max={v[-1]}")
    print("Targets: short <100, medium ~500, long >1000. If off, edit TARGET_WORDS in make_prompts.py "
          "and regenerate.")


def parent(a):
    models = find_models(a.models_dir, a.only)
    prompts = load_prompts(a.prompts_dir)
    if not models:
        sys.exit(f"No .gguf files found in {a.models_dir}/ (or none match --only).")
    if not prompts:
        sys.exit(f"No prompts in {a.prompts_dir}/. Run: python make_prompts.py")

    total = len(prompts) * len(models) * a.trials
    done = len(done_keys(a.out))
    print(f"Models : {', '.join(model_label(m) for m in models)}")
    print(f"Prompts: {len(prompts)}  Trials: {a.trials}  -> {total} timed runs ({done} already in {a.out})")
    print(f"Params : threads={a.threads} n_ctx={a.n_ctx} temp={a.temperature} max_tokens={a.max_tokens}")
    if a.dry_run:
        dry_run(models, prompts)
        return

    ensure_header(a.out)
    meta = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(), "machine": platform.machine(), "python": sys.version.split()[0],
        "threads": a.threads, "n_ctx": a.n_ctx, "temperature": a.temperature, "max_tokens": a.max_tokens,
        "trials": a.trials, "seed": a.seed, "models": [str(m) for m in models],
    }
    try:
        import llama_cpp
        meta["llama_cpp_python"] = llama_cpp.__version__
    except Exception:
        pass
    for key, cmd in (("cpu", ["sysctl", "-n", "machdep.cpu.brand_string"]),
                     ("mem_bytes", ["sysctl", "-n", "hw.memsize"]), ("macos", ["sw_vers", "-productVersion"])):
        try:
            meta[key] = subprocess.check_output(cmd).decode().strip()
        except Exception:
            pass
    Path("run_metadata.json").write_text(json.dumps(meta, indent=2))

    me = os.path.abspath(__file__)
    for idx, m in enumerate(models):
        cmd = [sys.executable, me, "--worker", "--model", str(m), "--prompts-dir", a.prompts_dir,
               "--out", a.out, "--outputs", a.outputs, "--trials", str(a.trials),
               "--threads", str(a.threads), "--n-ctx", str(a.n_ctx), "--max-tokens", str(a.max_tokens),
               "--temperature", str(a.temperature), "--seed", str(a.seed)]
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"!! {m.name} exited with code {rc} (out of memory?). Completed rows are kept; "
                  f"re-run to resume.", flush=True)
        if idx < len(models) - 1 and a.cooldown > 0:
            print(f"Cooling down {a.cooldown}s before next model...", flush=True)
            time.sleep(a.cooldown)
    print(f"\nDone. Rows in {a.out}: {len(done_keys(a.out))} / {total}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--prompts-dir", default="prompts")
    ap.add_argument("--out", default="benchmark_results.csv")
    ap.add_argument("--outputs", default="outputs.jsonl", help="generated text, for quality checks")
    ap.add_argument("--only", nargs="*", help="only models whose file name contains one of these")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--threads", type=int, default=physical_cores())
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cooldown", type=int, default=30, help="seconds between models")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--model", help=argparse.SUPPRESS)
    a = ap.parse_args()
    worker(a) if a.worker else parent(a)


if __name__ == "__main__":
    main()
