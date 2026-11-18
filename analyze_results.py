#!/usr/bin/env python3
"""Analyze benchmark_results.csv: Mean +/- StdDev tables, prefill-vs-decode attribution,
keyword effects, cold-cache outliers, and the 6 mandatory charts.

  python analyze_results.py                         # reads benchmark_results.csv
  python analyze_results.py --input partial.csv     # test on partial data
  python analyze_results.py --exclude-cold          # drop each model's first (cold-cache) run

Outputs: results/*.csv and figures/graph1..graph6 *.png (300 dpi).
Error bars are the standard deviation over all runs in the group (topics x trials).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_ORDER = ["TinyLlama-1.1B", "Qwen2.5-1.5B", "Phi-3-mini"]
CAT_ORDER = ["factual", "reasoning", "role_based", "instruction", "creative"]
LEN_ORDER = ["short", "medium", "long"]
KW_ORDER = ["none", "think_step", "expert_role", "concise", "detailed"]
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
PREFILL_C, DECODE_C = "#4C78A8", "#F58518"

plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 300, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "font.size": 10})


def ordered(values, preferred):
    present = [v for v in preferred if v in set(values)]
    return present + sorted(set(values) - set(present))


def ms(g, cols):
    """Mean and std (ddof=1, 0 when n=1) of cols."""
    a = g[cols].agg(["mean", "std", "count"])
    return a


def pm(mean, std, nd=1):
    return f"{mean:.{nd}f} ± {0 if pd.isna(std) else std:.{nd}f}"


def grouped_bar(ax, data, x, hue, y, x_order, hue_order, colors):
    g = data.groupby([x, hue])[y].agg(["mean", "std"])
    n = len(hue_order)
    w = 0.8 / n
    idx = np.arange(len(x_order))
    for i, h in enumerate(hue_order):
        means = [g["mean"].get((xv, h), np.nan) for xv in x_order]
        stds = [np.nan_to_num(g["std"].get((xv, h), np.nan)) for xv in x_order]
        ax.bar(idx - 0.4 + w / 2 + i * w, means, w, yerr=stds, capsize=3, label=h, color=colors[h])
    ax.set_xticks(idx)
    ax.set_xticklabels(x_order, rotation=0)


def save(fig, name):
    Path("figures").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"figures/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="benchmark_results.csv")
    ap.add_argument("--exclude-cold", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    if df.empty:
        raise SystemExit("CSV has no rows yet.")
    if a.exclude_cold:
        df = df[df["cold_start"] == 0]
    df["total_ms"] = df["t_prefill_ms"] + df["t_decode_ms"]
    df["prefill_share"] = df["t_prefill_ms"] / df["total_ms"]
    df["prefill_ms_per_token"] = df["t_prefill_ms"] / df["input_tokens"]
    df["ms_per_total_token"] = df["total_ms"] / df["total_tokens"]

    models = ordered(df["model"], MODEL_ORDER)
    cats = ordered(df["category"], CAT_ORDER)
    mcol = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}
    Path("results").mkdir(exist_ok=True)

    if (df["timing_source"] != "c_perf").any():
        n = int((df["timing_source"] != "c_perf").sum())
        print(f"NOTE: {n} rows used wall-clock fallback timers instead of llama.cpp C-level timers.")

    metrics = ["input_tokens", "output_tokens", "total_tokens", "t_prefill_ms", "t_decode_ms",
               "total_ms", "decode_tps", "ms_per_output_token", "peak_rss_mb", "kv_cache_est_mb"]

    # ---- tables (Mean ± StdDev) -------------------------------------------------------
    def table(keys, data, name):
        g = data.groupby(keys)[metrics].agg(["mean", "std", "count"])
        flat = g.copy()
        flat.columns = [f"{m}_{s}" for m, s in flat.columns]
        flat.reset_index().to_csv(f"results/{name}.csv", index=False)
        return g

    base = df[df["keyword"] == "none"]
    table(["model"], df, "summary_by_model")
    table(["model", "category"], base, "summary_by_category")
    table(["model", "length"], base, "summary_by_length")
    kw_short = df[df["length"] == "short"]
    table(["model", "keyword"], kw_short, "summary_by_keyword")

    print("\n=== Decode throughput and memory by model (Mean ± Std over all runs) ===")
    for m in models:
        d = df[df["model"] == m]
        print(f"{m:<16} decode {pm(d.decode_tps.mean(), d.decode_tps.std())} tok/s | "
              f"prefill {pm(d.t_prefill_ms.mean(), d.t_prefill_ms.std(), 0)} ms | "
              f"decode {pm(d.t_decode_ms.mean(), d.t_decode_ms.std(), 0)} ms | "
              f"peak RSS max {d.peak_rss_mb.max():.0f} MB | n={len(d)}")

    print("\n=== Median real input tokens per length class ===")
    print(df.groupby("length")["input_tokens"].median().reindex(LEN_ORDER).dropna().to_string())

    # ---- overhead attribution (prefill vs decode) -------------------------------------
    attr = (base.groupby(["model", "category", "length"])
            .agg(prefill_ms=("t_prefill_ms", "mean"), decode_ms=("t_decode_ms", "mean"),
                 prefill_share=("prefill_share", "mean"), input_tokens=("input_tokens", "mean"),
                 output_tokens=("output_tokens", "mean")).reset_index())
    attr.to_csv("results/overhead_attribution.csv", index=False)

    # ---- keyword effects vs baseline (same model/category/topic, short prompts) --------
    per = (kw_short.groupby(["model", "category", "topic", "keyword"])
           [["t_prefill_ms", "t_decode_ms", "total_ms", "output_tokens", "decode_tps"]].mean().reset_index())
    basel = per[per["keyword"] == "none"].drop(columns="keyword")
    mrg = per[per["keyword"] != "none"].merge(basel, on=["model", "category", "topic"], suffixes=("", "_base"))
    for c in ["t_prefill_ms", "t_decode_ms", "total_ms", "output_tokens", "decode_tps"]:
        mrg[f"{c}_pct"] = (mrg[c] / mrg[f"{c}_base"] - 1) * 100
    kw_eff = (mrg.groupby(["model", "keyword"])[[c for c in mrg.columns if c.endswith("_pct")]]
              .mean().reset_index())
    kw_eff.to_csv("results/keyword_effects_pct_vs_base.csv", index=False)
    print("\n=== Keyword effect, % change vs. same prompt without the keyword (short prompts) ===")
    print(kw_eff.round(1).to_string(index=False))

    # ---- cold-cache / throttling outliers ------------------------------------------------
    med = df.groupby(["model", "prompt_id"])["total_ms"].transform("median")
    out = df[(df["total_ms"] > 1.5 * med) & (df.groupby(["model", "prompt_id"])["trial"].transform("count") > 1)]
    out[["model", "prompt_id", "trial", "cold_start", "total_ms"]].assign(
        median_total_ms=med[out.index]).to_csv("results/outliers.csv", index=False)
    print(f"\nOutliers (>1.5x the median of the 3 trials): {len(out)} rows "
          f"({int(out.cold_start.sum())} are a model's first cold-cache run) -> results/outliers.csv")

    # ---- the 6 charts ---------------------------------------------------------------------
    print("\nCharts:")
    # 1. decode tokens/s by model
    fig, ax = plt.subplots(figsize=(6, 4.2))
    g = df.groupby("model")["decode_tps"].agg(["mean", "std"]).reindex(models).fillna(0)
    ax.bar(models, g["mean"], yerr=g["std"], capsize=4, color=[mcol[m] for m in models])
    for i, v in enumerate(g["mean"]):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    ax.set_ylabel("Decode throughput (tokens/s)")
    ax.set_title("Graph 1: Decode throughput by model (mean ± std)")
    save(fig, "graph1_decode_tps_by_model")

    # 2. prefill vs decode latency by category (one panel per model)
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.4), sharey=False, squeeze=False)
    for ax, m in zip(axes[0], models):
        bs = base[(base["model"] == m) & (base["length"] == "short")]
        d = bs.groupby("category")[["t_prefill_ms", "t_decode_ms"]].mean().reindex(cats)
        s = bs.groupby("category")["total_ms"].std().reindex(cats).fillna(0)
        ax.bar(cats, d["t_prefill_ms"], label="Prefill", color=PREFILL_C)
        ax.bar(cats, d["t_decode_ms"], bottom=d["t_prefill_ms"], yerr=s, capsize=3, label="Decode", color=DECODE_C)
        ax.set_title(m)
        ax.set_ylabel("Latency (ms)")
        ax.tick_params(axis="x", rotation=30)
    axes[0][0].legend()
    fig.suptitle("Graph 2: Prefill vs decode latency by prompt category (short prompts, no keyword prefix)")
    save(fig, "graph2_prefill_vs_decode_by_category")

    # 3. peak RSS by model (grouped by length class, since RSS is a running peak)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    lens = ordered(base["length"], LEN_ORDER)
    grouped_bar(ax, base, "length", "model", "peak_rss_mb", lens, models, mcol)
    ax.set_ylabel("Peak resident memory, ru_maxrss (MB)")
    ax.set_xlabel("Prompt length class")
    ax.set_title("Graph 3: Peak memory (RSS) by model")
    ax.legend()
    save(fig, "graph3_peak_memory_by_model")

    # 4. prompt length vs latency
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, col, title in ((axes[0], "total_ms", "Total latency"), (axes[1], "t_prefill_ms", "Prefill latency")):
        for m in models:
            g = base[base["model"] == m].groupby("length")[col].agg(["mean", "std"]).reindex(lens).fillna(0)
            ax.errorbar(lens, g["mean"], yerr=g["std"], marker="o", capsize=3, label=m, color=mcol[m])
        ax.set_ylabel(f"{title} (ms)")
        ax.set_xlabel("Prompt length class (<100, ~500, >1000 tokens)")
        ax.set_title(title)
    axes[0].legend()
    fig.suptitle("Graph 4: Prompt sequence length vs latency")
    save(fig, "graph4_length_vs_latency")

    # 5. normalized ms/token across categories
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    grouped_bar(axes[0], base, "category", "model", "ms_per_output_token", cats, models, mcol)
    axes[0].set_ylabel("Decode cost (ms per generated token)")
    axes[0].set_title("Decode: ms / output token")
    grouped_bar(axes[1], base, "category", "model", "prefill_ms_per_token", cats, models, mcol)
    axes[1].set_ylabel("Prefill cost (ms per prompt token)")
    axes[1].set_title("Prefill: ms / input token")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    axes[0].legend()
    fig.suptitle("Graph 5: Normalized per-token cost by prompt category")
    save(fig, "graph5_ms_per_token_by_category")

    # 6. keyword effect comparison (short prompts)
    kws = ordered(kw_short["keyword"], KW_ORDER)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, col, lab in ((axes[0], "t_prefill_ms", "Prefill latency (ms)"),
                         (axes[1], "output_tokens", "Output tokens generated"),
                         (axes[2], "total_ms", "Total latency (ms)")):
        grouped_bar(ax, kw_short, "keyword", "model", col, kws, models, mcol)
        ax.set_ylabel(lab)
        ax.tick_params(axis="x", rotation=25)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(models), frameon=False)
    fig.suptitle("Graph 6: Prompt keyword effect (short prompts; none = base question)")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/graph6_keyword_effects.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/graph6_keyword_effects.png")

    print("\nTables are in results/. Open the charts with:  open figures")


if __name__ == "__main__":
    main()
