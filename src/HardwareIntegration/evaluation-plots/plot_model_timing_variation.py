#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_common import (
    DATA_ROOT,
    MODEL_TRACES,
    OUT_ROOT,
    PALETTE,
    apply_style,
    boxplot_by_label,
    describe,
    load_llm_behavior,
    save_figure,
    write_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot timing variation across LLM models.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT / "model-timing-variation")
    parser.add_argument(
        "--maxwait-ms",
        type=float,
        default=None,
        help="Draw this maxwait value. Defaults to the largest measured worst-case inference latency.",
    )
    args = parser.parse_args()

    apply_style()
    llm_df = pd.concat(
        [load_llm_behavior(args.data_root / dirname, label) for label, dirname in MODEL_TRACES.items()],
        ignore_index=True,
    )

    rows = []
    for label in MODEL_TRACES:
        subset = llm_df[llm_df["condition"] == label]
        rows.append({"model": label, "metric": "inference_ms", **describe(subset["inference_ms"])})
        rows.append({"model": label, "metric": "lag_ms", **describe(subset["lag_ms"])})
    write_summary(args.out_dir / "model_timing_variation_summary.csv", rows)

    maxwait = args.maxwait_ms
    if maxwait is None:
        maxwait = float(llm_df["inference_ms"].max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    boxplot_by_label(
        axes[0],
        {label: llm_df.loc[llm_df["condition"] == label, "inference_ms"] for label in MODEL_TRACES},
        "Inference latency (ms)",
    )
    axes[0].axhline(maxwait, color="#111827", linestyle="--", linewidth=1.1, label=f"maxwait = {maxwait:.1f} ms")
    axes[0].set_title("Inference timing by model")
    axes[0].legend()

    summary = pd.DataFrame(rows)
    inference = summary[summary["metric"] == "inference_ms"].set_index("model")
    x = range(len(inference.index))
    axes[1].bar(
        [i - 0.18 for i in x],
        inference["mean"],
        width=0.36,
        label="Mean",
        color=[PALETTE.get(label, "#64748b") for label in inference.index],
        alpha=0.65,
    )
    axes[1].bar([i + 0.18 for i in x], inference["max"], width=0.36, label="Worst case", color="#111827", alpha=0.75)
    axes[1].axhline(maxwait, color="#dc2626", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(list(x), list(inference.index))
    axes[1].set_ylabel("Inference latency (ms)")
    axes[1].set_title("Mean vs worst-case latency")
    axes[1].legend()

    save_figure(fig, args.out_dir, "model_timing_variation")
    print(f"Wrote {args.out_dir / 'model_timing_variation.png'}")
    print(f"Wrote {args.out_dir / 'model_timing_variation_summary.csv'}")


if __name__ == "__main__":
    main()
