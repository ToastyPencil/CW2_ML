#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cw2_ml.experiments.run_active_learning import ActiveLearningConfig, run_experiments  # noqa: E402
from cw2_ml.utils.io import ensure_dir  # noqa: E402
from scripts.fill_report_from_outputs import main as fill_report_main  # noqa: E402
from scripts.generate_report_artifacts import main as generate_report_main  # noqa: E402


DEFAULT_SCREENING_GRID: list[tuple[float, int]] = [
    (0.05, 2),
    (0.10, 2),
    (0.20, 2),
    (0.10, 3),
    (0.15, 3),
    (0.20, 3),
    (0.25, 4),
]


def candidate_slug(novelty_weight: float, novelty_start_round: int) -> str:
    weight_text = f"{novelty_weight:.2f}".replace(".", "p")
    return f"w{weight_text}_s{novelty_start_round}"


def summarize_candidate_metrics(
    metrics_path: Path,
    novelty_weight: float,
    novelty_start_round: int,
) -> dict[str, float | int | str]:
    df = pd.read_csv(metrics_path)
    final_round = int(df["round"].max())
    final_df = df[df["round"] == final_round]
    return {
        "candidate": candidate_slug(novelty_weight, novelty_start_round),
        "novelty_weight": float(novelty_weight),
        "novelty_start_round": int(novelty_start_round),
        "final_round": final_round,
        "final_round_mean_accuracy": round(float(final_df["test_accuracy"].mean()), 10),
        "final_round_std_accuracy": round(float(final_df["test_accuracy"].std(ddof=1) if len(final_df) > 1 else 0.0), 10),
        "mean_accuracy_across_rounds": round(float(df["test_accuracy"].mean()), 10),
    }


def rank_candidate_summaries(summary_df: pd.DataFrame) -> pd.DataFrame:
    return summary_df.sort_values(
        ["final_round_mean_accuracy", "mean_accuracy_across_rounds", "final_round_std_accuracy", "candidate"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def prefer_candidate_over_reference(
    candidate_summary: dict[str, float | int | str],
    reference_summary: dict[str, float | int | str],
) -> bool:
    candidate_final = float(candidate_summary["final_round_mean_accuracy"])
    reference_final = float(reference_summary["final_round_mean_accuracy"])
    if candidate_final != reference_final:
        return candidate_final > reference_final

    candidate_mean = float(candidate_summary["mean_accuracy_across_rounds"])
    reference_mean = float(reference_summary["mean_accuracy_across_rounds"])
    if candidate_mean != reference_mean:
        return candidate_mean > reference_mean

    return float(candidate_summary["final_round_std_accuracy"]) <= float(reference_summary["final_round_std_accuracy"])


def _save_outputs(df: pd.DataFrame, output_dir: Path) -> Path:
    out_dir = ensure_dir(output_dir)
    metrics_path = out_dir / "metrics.csv"
    df.to_csv(metrics_path, index=False)
    if not df.empty:
        summary = (
            df.groupby(["strategy", "round"], as_index=False)
            .agg(
                labeled_count_mean=("labeled_count", "mean"),
                accuracy_mean=("test_accuracy", "mean"),
                accuracy_std=("test_accuracy", "std"),
            )
            .fillna(0.0)
        )
        summary.to_csv(out_dir / "summary.csv", index=False)
    return metrics_path


def _run_candidate(
    *,
    data_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    repeats: int,
    rounds: int,
    query_batch_size: int,
    train_epochs: int,
    device: str,
    novelty_weight: float,
    novelty_start_round: int,
) -> dict[str, float | int | str]:
    metrics_path = output_dir / "metrics.csv"
    if not metrics_path.exists():
        cfg = ActiveLearningConfig(
            strategy="typiclust_adaptive",
            data_dir=str(data_dir),
            output_dir=str(output_dir),
            repeats=repeats,
            rounds=rounds,
            query_batch_size=query_batch_size,
            train_epochs=train_epochs,
            device=device,
            novelty_weight=novelty_weight,
            novelty_start_round=novelty_start_round,
            ssl_pretrain_epochs=500,
            ssl_checkpoint_path=str(cache_dir / "simclr_resnet18.pt"),
            ssl_embeddings_path=str(cache_dir / "cifar10_embeddings.npy"),
        )
        df = run_experiments(config=cfg, strategies=["typiclust_adaptive"])
        _save_outputs(df, output_dir=output_dir)
    return summarize_candidate_metrics(
        metrics_path=metrics_path,
        novelty_weight=novelty_weight,
        novelty_start_round=novelty_start_round,
    )


def _merge_candidate_with_reference(
    *,
    reference_metrics_path: Path,
    candidate_metrics_path: Path,
    output_dir: Path,
) -> Path:
    reference_df = pd.read_csv(reference_metrics_path)
    candidate_df = pd.read_csv(candidate_metrics_path)
    merged_df = pd.concat(
        [
            reference_df[reference_df["strategy"] != "typiclust_adaptive"],
            candidate_df,
        ],
        ignore_index=True,
    ).sort_values(["strategy", "repeat", "round"])
    return _save_outputs(merged_df, output_dir=output_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune TypiClust+Adaptive while reusing cached SSL artifacts.")
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
    parser.add_argument("--reference-output-dir", type=str, default=str(ROOT / "outputs/final_submission_500"))
    parser.add_argument("--output-root", type=str, default=str(ROOT / "outputs/tuned_adaptive"))
    parser.add_argument("--report-dir", type=str, default=str(ROOT / "report"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--query-batch-size", type=int, default=10)
    parser.add_argument("--train-epochs", type=int, default=100)
    parser.add_argument("--screen-repeats", type=int, default=1)
    parser.add_argument("--confirm-repeats", type=int, default=3)
    parser.add_argument(
        "--grid",
        nargs="*",
        default=[],
        help="Optional candidate grid entries formatted as weight:start, e.g. 0.10:3 0.20:4",
    )
    return parser.parse_args()


def _parse_grid(raw_grid: list[str]) -> list[tuple[float, int]]:
    if not raw_grid:
        return DEFAULT_SCREENING_GRID

    candidates: list[tuple[float, int]] = []
    for item in raw_grid:
        weight_text, start_text = item.split(":")
        candidates.append((float(weight_text), int(start_text)))
    return candidates


def main() -> None:
    args = _parse_args()
    data_dir = Path(args.data_dir)
    reference_output_dir = Path(args.reference_output_dir)
    cache_dir = reference_output_dir / "ssl"
    output_root = ensure_dir(args.output_root)
    screen_dir = ensure_dir(output_root / "screen")
    confirm_dir = ensure_dir(output_root / "confirm")
    merged_output_dir = ensure_dir(output_root / "final_submission_tuned")
    report_artifacts_dir = ensure_dir(output_root / "report_artifacts_tuned")
    candidates = _parse_grid(args.grid)

    screen_summaries: list[dict[str, float | int | str]] = []
    for novelty_weight, novelty_start_round in candidates:
        slug = candidate_slug(novelty_weight, novelty_start_round)
        print(f"[screen] {slug}", flush=True)
        screen_summaries.append(
            _run_candidate(
                data_dir=data_dir,
                cache_dir=cache_dir,
                output_dir=screen_dir / slug,
                repeats=args.screen_repeats,
                rounds=args.rounds,
                query_batch_size=args.query_batch_size,
                train_epochs=args.train_epochs,
                device=args.device,
                novelty_weight=novelty_weight,
                novelty_start_round=novelty_start_round,
            )
        )

    screen_summary_df = rank_candidate_summaries(pd.DataFrame(screen_summaries))
    screen_summary_path = output_root / "screening_summary.csv"
    screen_summary_df.to_csv(screen_summary_path, index=False)
    best_candidate = screen_summary_df.iloc[0]
    best_weight = float(best_candidate["novelty_weight"])
    best_start_round = int(best_candidate["novelty_start_round"])
    best_slug = str(best_candidate["candidate"])
    print(f"[confirm] {best_slug}", flush=True)

    confirm_summary = _run_candidate(
        data_dir=data_dir,
        cache_dir=cache_dir,
        output_dir=confirm_dir / best_slug,
        repeats=args.confirm_repeats,
        rounds=args.rounds,
        query_batch_size=args.query_batch_size,
        train_epochs=args.train_epochs,
        device=args.device,
        novelty_weight=best_weight,
        novelty_start_round=best_start_round,
    )

    confirm_summary_path = output_root / "confirm_summary.csv"
    pd.DataFrame([confirm_summary]).to_csv(confirm_summary_path, index=False)

    reference_adaptive_path = output_root / "reference_adaptive_metrics.csv"
    reference_df = pd.read_csv(reference_output_dir / "metrics.csv")
    reference_adaptive_df = reference_df[reference_df["strategy"] == "typiclust_adaptive"].copy()
    reference_adaptive_df.to_csv(reference_adaptive_path, index=False)
    reference_summary = summarize_candidate_metrics(
        reference_adaptive_path,
        novelty_weight=0.20,
        novelty_start_round=2,
    )

    selected_candidate_metrics_path = confirm_dir / best_slug / "metrics.csv"
    selected_summary = confirm_summary
    if not prefer_candidate_over_reference(confirm_summary, reference_summary):
        selected_candidate_metrics_path = reference_adaptive_path
        selected_summary = reference_summary
        pd.DataFrame([selected_summary]).to_csv(output_root / "selected_summary.csv", index=False)
    else:
        pd.DataFrame([selected_summary]).to_csv(output_root / "selected_summary.csv", index=False)

    merged_metrics_path = _merge_candidate_with_reference(
        reference_metrics_path=reference_output_dir / "metrics.csv",
        candidate_metrics_path=selected_candidate_metrics_path,
        output_dir=merged_output_dir,
    )

    generate_report_main(
        [
            "--input",
            str(merged_metrics_path),
            "--output",
            str(report_artifacts_dir),
            "--baseline",
            "random",
        ]
    )
    fill_report_main(
        [
            "--artifacts-dir",
            str(report_artifacts_dir),
            "--report-dir",
            args.report_dir,
            "--pretrain-epochs",
            "500",
            "--gpu-name",
            "RTX 3070",
        ]
    )
    print(f"Saved tuning summaries to {screen_summary_path} and {confirm_summary_path}", flush=True)
    print(f"Saved merged metrics to {merged_metrics_path}", flush=True)


if __name__ == "__main__":
    main()
