import warnings
from pathlib import Path

import pandas as pd

from scripts.generate_report_artifacts import main


def test_generate_report_artifacts_writes_expected_files(tmp_path: Path) -> None:
    metrics = pd.DataFrame(
        {
            "strategy": ["random", "random", "typiclust", "typiclust_adaptive"],
            "repeat": [0, 1, 0, 0],
            "round": [5, 5, 5, 5],
            "labeled_count": [50, 50, 50, 50],
            "test_accuracy": [0.51, 0.53, 0.60, 0.64],
        }
    )
    metrics_path = tmp_path / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    output_dir = tmp_path / "artifacts"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        main(
            [
                "--input",
                str(metrics_path),
                "--output",
                str(output_dir),
                "--baseline",
                "random",
            ]
        )

    assert (output_dir / "summary_by_round.csv").exists()
    assert (output_dir / "statistical_comparisons.csv").exists()
    assert (output_dir / "accuracy_vs_round.png").exists()
    assert not caught

    comparisons = pd.read_csv(output_dir / "statistical_comparisons.csv")
    adaptive_vs_typiclust = comparisons[
        (comparisons["strategy_a"] == "typiclust_adaptive")
        & (comparisons["strategy_b"] == "typiclust")
    ]
    assert not adaptive_vs_typiclust.empty
