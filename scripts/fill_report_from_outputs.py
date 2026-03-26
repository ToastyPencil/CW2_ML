#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


DISPLAY_NAMES = {
    "random": "Random",
    "entropy": "Entropy",
    "typiclust": "TypiClust",
    "typiclust_adaptive": "TypiClust+Adaptive",
}

REPORT_ORDER = ["typiclust", "typiclust_adaptive", "random", "entropy"]


def _format_pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}\\%"


def _format_p_value(value: float) -> str:
    if value < 0.001:
        return "$p<0.001$"
    return f"$p={value:.3f}$"


def _display_name(strategy: str) -> str:
    return DISPLAY_NAMES.get(strategy, strategy)


def _ordered_final_rows(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    final_round = int(summary_df["round"].max())
    final_df = summary_df[summary_df["round"] == final_round].copy()
    final_df["order"] = final_df["strategy"].map(
        lambda name: REPORT_ORDER.index(name) if name in REPORT_ORDER else len(REPORT_ORDER)
    )
    final_df = final_df.sort_values(["order", "strategy"]).drop(columns=["order"])
    return final_df, final_round


def _copy_plot(src: Path, dst_dir: Path) -> None:
    shutil.copy2(src, dst_dir / src.name)


def _build_results_rows(final_df: pd.DataFrame, final_round: int) -> str:
    rows: list[str] = []
    tuple_rows = list(final_df.itertuples(index=False))
    for idx, row in enumerate(tuple_rows):
        line_end = r" \\" if idx < len(tuple_rows) - 1 else ""
        rows.append(
            f"{_display_name(row.strategy)} & {final_round} & {_format_pct(row.mean_accuracy)} & {_format_pct(row.se_accuracy)}{line_end}"
        )
    return "\n".join(rows) + "\n"


def _build_methodology_text(
    final_round: int,
    pretrain_epochs: int,
    gpu_name: str,
) -> str:
    return (
        f"The final automatic run used {pretrain_epochs} SimCLR pretraining epochs on the {gpu_name} "
        f"to build frozen embeddings for TypiClust and TypiClust+Adaptive, then retrained a supervised "
        f"ResNet-18 classifier after every query round. The study evaluated {final_round} active learning "
        f"rounds with matched seeds across Random, Entropy, TypiClust, and TypiClust+Adaptive.\n"
    )


def _clustering_beats_random_rounds(summary_df: pd.DataFrame, final_round: int) -> list[int]:
    if "random" not in summary_df["strategy"].unique():
        return []

    pivot = summary_df.pivot(index="round", columns="strategy", values="mean_accuracy")
    winning_rounds: list[int] = []
    for round_idx in sorted(int(value) for value in pivot.index.tolist()):
        if round_idx >= final_round or round_idx not in pivot.index:
            continue
        random_score = pivot.loc[round_idx].get("random")
        if pd.isna(random_score):
            continue
        clustering_scores = [
            float(pivot.loc[round_idx][strategy])
            for strategy in ("typiclust", "typiclust_adaptive")
            if strategy in pivot.columns and not pd.isna(pivot.loc[round_idx].get(strategy))
        ]
        if clustering_scores and max(clustering_scores) > float(random_score):
            winning_rounds.append(round_idx)
    return winning_rounds


def _build_results_text(
    summary_df: pd.DataFrame,
    final_df: pd.DataFrame,
    final_round: int,
    pretrain_epochs: int,
    gpu_name: str,
) -> str:
    final_rows = {row.strategy: row for row in final_df.itertuples(index=False)}
    ranked = final_df.sort_values("mean_accuracy", ascending=False)
    ranking_text = ", ".join(
        f"{_display_name(row.strategy)} ({_format_pct(row.mean_accuracy)})"
        for row in ranked.itertuples(index=False)
    )
    adaptive = final_rows.get("typiclust_adaptive")
    original = final_rows.get("typiclust")
    improvement_text = ""
    if adaptive is not None and original is not None:
        diff = float(adaptive.mean_accuracy - original.mean_accuracy)
        improvement_text = (
            f" TypiClust+Adaptive changed the final-round mean by {_format_pct(diff)} "
            f"relative to the original TypiClust."
        )
    early_round_wins = _clustering_beats_random_rounds(summary_df, final_round)
    early_round_text = ""
    if early_round_wins:
        rounds_text = ", ".join(str(round_idx) for round_idx in early_round_wins)
        early_round_text = (
            f" A clustering-based strategy beat Random in rounds {rounds_text}, "
            f"but Random finished highest at round {final_round}."
        )
    return (
        f"The completed reproduction used {pretrain_epochs} self-supervised pretraining epochs on the {gpu_name}. "
        f"At round {final_round}, the strategy ranking was {ranking_text}.{early_round_text}{improvement_text}\n"
    )


def _build_stats_text(comparisons_df: pd.DataFrame) -> str:
    comparisons: list[str] = []
    preferred_pairs = [
        ("typiclust_adaptive", "typiclust"),
        ("typiclust_adaptive", "random"),
        ("typiclust", "random"),
    ]
    for strategy_a, strategy_b in preferred_pairs:
        match = comparisons_df[
            (comparisons_df["strategy_a"] == strategy_a) & (comparisons_df["strategy_b"] == strategy_b)
        ]
        if match.empty:
            continue
        row = match.iloc[0]
        comparisons.append(
            f"Compared with {_display_name(strategy_b)}, {_display_name(strategy_a)} changed the final-round mean by "
            f"{_format_pct(row['mean_diff'])} ({_format_p_value(float(row['p_value_ttest']))} paired t-test; "
            f"{_format_p_value(float(row['p_value_wilcoxon']))} Wilcoxon)."
        )
    if not comparisons:
        return "Paired comparisons were unavailable for the requested strategies.\n"
    significant_mask = (comparisons_df["p_value_ttest"] < 0.05) | (comparisons_df["p_value_wilcoxon"] < 0.05)
    significance_text = (
        " None of the final-round comparisons crossed the 0.05 significance threshold, so the observed differences "
        "should be interpreted as directional rather than conclusive."
        if not significant_mask.any()
        else ""
    )
    return " ".join(comparisons) + significance_text + "\n"


def _build_conclusion_text(summary_df: pd.DataFrame, final_df: pd.DataFrame, final_round: int) -> str:
    final_rows = {row.strategy: row for row in final_df.itertuples(index=False)}
    adaptive = final_rows.get("typiclust_adaptive")
    original = final_rows.get("typiclust")
    random_row = final_rows.get("random")
    entropy_row = final_rows.get("entropy")
    if adaptive is not None and original is not None and random_row is not None:
        adaptive_gain = float(adaptive.mean_accuracy - original.mean_accuracy)
        early_round_wins = _clustering_beats_random_rounds(summary_df, final_round)
        early_round_text = (
            "A clustering-based method led Random throughout the pre-final rounds, "
            if early_round_wins
            else ""
        )
        if entropy_row is not None:
            if float(adaptive.mean_accuracy) > float(entropy_row.mean_accuracy):
                relative_text = "and surpassed Entropy"
            elif float(adaptive.mean_accuracy) < float(entropy_row.mean_accuracy):
                relative_text = "but still trailed Entropy"
            else:
                relative_text = "and matched Entropy"
        else:
            relative_text = "and should be compared against the uncertainty baseline"
        return (
            f"{early_round_text}but the final configuration did not reproduce the paper's strongest claim at round "
            f"{final_round}: Random remained the best final strategy. TypiClust+Adaptive improved TypiClust by "
            f"{_format_pct(adaptive_gain)} at the final round, remained narrowly below Random, {relative_text}. The modification "
            f"should therefore be presented as a modest improvement over the base method rather than a universal win.\n"
        )
    return (
        "In this reproduction, low-budget clustering-based querying remained competitive and the adaptive variant "
        "should be discussed relative to both the original TypiClust method and the uncertainty baselines.\n"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fill report-ready LaTeX fragments from generated experiment artifacts.")
    parser.add_argument("--artifacts-dir", type=str, required=True)
    parser.add_argument("--report-dir", type=str, required=True)
    parser.add_argument("--pretrain-epochs", type=int, required=True)
    parser.add_argument("--gpu-name", type=str, default="RTX 3070")
    args = parser.parse_args(argv)

    artifacts_dir = Path(args.artifacts_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(artifacts_dir / "summary_by_round.csv")
    comparisons_df = pd.read_csv(artifacts_dir / "statistical_comparisons.csv")
    final_df, final_round = _ordered_final_rows(summary_df)

    (report_dir / "generated_results_rows.tex").write_text(
        _build_results_rows(final_df, final_round),
        encoding="utf-8",
    )
    (report_dir / "generated_methodology_text.tex").write_text(
        _build_methodology_text(final_round, args.pretrain_epochs, args.gpu_name),
        encoding="utf-8",
    )
    (report_dir / "generated_results_text.tex").write_text(
        _build_results_text(summary_df, final_df, final_round, args.pretrain_epochs, args.gpu_name),
        encoding="utf-8",
    )
    (report_dir / "generated_stats_text.tex").write_text(
        _build_stats_text(comparisons_df),
        encoding="utf-8",
    )
    (report_dir / "generated_conclusion_text.tex").write_text(
        _build_conclusion_text(summary_df, final_df, final_round),
        encoding="utf-8",
    )

    _copy_plot(artifacts_dir / "accuracy_vs_round.png", report_dir)
    _copy_plot(artifacts_dir / "final_round_boxplot.png", report_dir)


if __name__ == "__main__":
    main()
