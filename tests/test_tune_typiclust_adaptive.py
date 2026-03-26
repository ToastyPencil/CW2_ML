from pathlib import Path
import subprocess
import sys

import pandas as pd

from scripts.tune_typiclust_adaptive import (
    DEFAULT_SCREENING_GRID,
    candidate_slug,
    prefer_candidate_over_reference,
    rank_candidate_summaries,
    summarize_candidate_metrics,
)


def test_candidate_slug_is_filesystem_friendly() -> None:
    assert candidate_slug(0.15, 3) == "w0p15_s3"


def test_summarize_candidate_metrics_extracts_final_and_average_scores(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    pd.DataFrame(
        {
            "strategy": ["typiclust_adaptive"] * 6,
            "repeat": [0, 0, 0, 1, 1, 1],
            "round": [1, 2, 3, 1, 2, 3],
            "labeled_count": [10, 20, 30, 10, 20, 30],
            "test_accuracy": [0.20, 0.30, 0.40, 0.25, 0.35, 0.45],
        }
    ).to_csv(metrics_path, index=False)

    summary = summarize_candidate_metrics(metrics_path, novelty_weight=0.15, novelty_start_round=3)

    assert summary["candidate"] == "w0p15_s3"
    assert summary["final_round"] == 3
    assert summary["final_round_mean_accuracy"] == 0.425
    assert summary["mean_accuracy_across_rounds"] == 0.325


def test_rank_candidate_summaries_prefers_final_round_then_learning_curve_mean() -> None:
    ranking = rank_candidate_summaries(
        pd.DataFrame(
            [
                {
                    "candidate": "a",
                    "final_round_mean_accuracy": 0.30,
                    "mean_accuracy_across_rounds": 0.20,
                    "final_round_std_accuracy": 0.01,
                },
                {
                    "candidate": "b",
                    "final_round_mean_accuracy": 0.30,
                    "mean_accuracy_across_rounds": 0.25,
                    "final_round_std_accuracy": 0.03,
                },
                {
                    "candidate": "c",
                    "final_round_mean_accuracy": 0.28,
                    "mean_accuracy_across_rounds": 0.27,
                    "final_round_std_accuracy": 0.01,
                },
            ]
        )
    )

    assert ranking["candidate"].tolist() == ["b", "a", "c"]


def test_default_screening_grid_includes_current_coursework_setting() -> None:
    assert (0.20, 2) in DEFAULT_SCREENING_GRID


def test_tuning_script_runs_help_as_direct_script() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/tune_typiclust_adaptive.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Tune TypiClust+Adaptive" in result.stdout


def test_prefer_candidate_over_reference_requires_nonworse_final_round() -> None:
    candidate = {
        "final_round_mean_accuracy": 0.21,
        "mean_accuracy_across_rounds": 0.18,
        "final_round_std_accuracy": 0.02,
    }
    reference = {
        "final_round_mean_accuracy": 0.20,
        "mean_accuracy_across_rounds": 0.19,
        "final_round_std_accuracy": 0.01,
    }

    assert prefer_candidate_over_reference(candidate, reference) is True


def test_prefer_candidate_over_reference_keeps_reference_when_candidate_regresses() -> None:
    candidate = {
        "final_round_mean_accuracy": 0.19,
        "mean_accuracy_across_rounds": 0.25,
        "final_round_std_accuracy": 0.01,
    }
    reference = {
        "final_round_mean_accuracy": 0.20,
        "mean_accuracy_across_rounds": 0.15,
        "final_round_std_accuracy": 0.02,
    }

    assert prefer_candidate_over_reference(candidate, reference) is False
