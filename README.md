# Prompt Keyword Effects on LLM Decode Efficiency (Topic 2)

Measures how prompt wording (keywords such as "Think step by step"), prompt category, and prompt
length change prefill latency, decode throughput, and memory for three small quantized LLMs,
running on an Intel MacBook Pro (CPU only) with llama.cpp via `llama-cpp-python`.

## Pipeline

    make_prompts.py  ->  prompts/*.json
    benchmark_runner.py  ->  benchmark_results.csv (+ outputs.jsonl, run_metadata.json)
    analyze_results.py   ->  results/*.csv, figures/graph1..6

## Setup (macOS, Intel)

    xcode-select --install
    brew install cmake python@3.11
    python3.11 -m venv .venv && source .venv/bin/activate
    CMAKE_ARGS="-DGGML_METAL=off -DGGML_AVX2=on -DGGML_FMA=on -DGGML_F16C=on" \
      pip install llama-cpp-python --no-cache-dir --force-reinstall
    pip install -r requirements.txt

Models (Q4_K_M GGUF, put in `models/`; not committed): TinyLlama-1.1B-Chat, Qwen2.5-1.5B-Instruct,
Phi-3-mini-4k-instruct.

## Run

    python make_prompts.py                 # or --topics-per-category 1 for a faster run
    python benchmark_runner.py --dry-run   # check run count and real token lengths
    caffeinate -dimsu python benchmark_runner.py 2>&1 | tee run.log
    python analyze_results.py

Controlled parameters: temperature 0.2, max 128 new tokens, context 2048, fixed seed and thread
count, 3 trials per prompt. Each model runs in its own process so peak RSS is per model.
The runner resumes if interrupted.

## Metrics

Input/output/total tokens; prefill latency (`t_p_eval_ms`); decode latency (`t_eval_ms`);
decode tokens/s; ms per generated token; peak RSS (`ru_maxrss`, MB); estimated KV-cache size.

## Limitations

Single machine and runtime; no energy, thermal or GPU/NPU measurement; N = 3 trials; output capped
at 128 tokens; output quality not scored (generated text is saved in `outputs.jsonl`).

## Hardware used

Fill in from `run_metadata.json` before submitting.
