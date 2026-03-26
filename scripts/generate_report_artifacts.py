#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cw2_ml.reporting.analysis import final_round_table, paired_summary, summarize_by_round
from cw2_ml.reporting.plots import plot_accuracy_by_round, plot_final_round_boxplot
from cw2_ml.utils.io import ensure_dir


def _resolve_metrics_path(input_path: Path) -> Path:
    if input_path.is_dir():
        return input_path / "metrics.csv"
    return input_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate tables/plots for the coursework report.")
    parser.add_argument("--input", type=str, required=True, help="Path to metrics.csv or its directory")
    parser.add_argument("--output", type=str, required=True, help="Directory for generated artifacts")
    parser.add_argument("--baseline", type=str, default="random", help="Baseline strategy for significance tests")
    args = parser.parse_args(argv)

    metrics_path = _resolve_metrics_path(Path(args.input))
    if not metrics_path.exists():
        raise FileNotFoundError(f"Could not find metrics file: {metrics_path}")

    output_dir = ensure_dir(args.output)
    df = pd.read_csv(metrics_path)

    summary = summarize_by_round(df)
    summary.to_csv(output_dir / "summary_by_round.csv", index=False)

    final_df = final_round_table(df)
    final_df.to_csv(output_dir / "final_round_raw.csv", index=False)

    strategies = sorted(df["strategy"].unique().tolist())
    comparisons = []
    added_pairs: set[tuple[str, str]] = set()

    def _append_pair(strategy_a: str, strategy_b: str) -> None:
        pair = (strategy_a, strategy_b)
        if pair in added_pairs:
            return
        try:
            comparisons.append(paired_summary(final_df, strategy_a=strategy_a, strategy_b=strategy_b))
            added_pairs.add(pair)
        except ValueError:
            return

    for strategy in strategies:
        if strategy == args.baseline:
            continue
        _append_pair(strategy, args.baseline)
    if {"typiclust", "typiclust_adaptive"} <= set(strategies):
        _append_pair("typiclust_adaptive", "typiclust")
    pd.DataFrame(comparisons).to_csv(output_dir / "statistical_comparisons.csv", index=False)

    plot_accuracy_by_round(df, output_dir / "accuracy_vs_round.png")
    plot_final_round_boxplot(df, output_dir / "final_round_boxplot.png")

    print(f"Saved report artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
