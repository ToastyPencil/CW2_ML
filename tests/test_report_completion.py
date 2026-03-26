from pathlib import Path

import pandas as pd

from scripts.fill_report_from_outputs import main


def test_fill_report_from_outputs_writes_generated_sections(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    report_dir = tmp_path / "report"
    artifacts_dir.mkdir()
    report_dir.mkdir()

    summary = pd.DataFrame(
        {
            "strategy": ["random", "typiclust", "typiclust_adaptive"],
            "round": [5, 5, 5],
            "mean_accuracy": [0.52, 0.61, 0.66],
            "std_accuracy": [0.01, 0.02, 0.015],
            "se_accuracy": [0.007, 0.014, 0.011],
        }
    )
    summary.to_csv(artifacts_dir / "summary_by_round.csv", index=False)

    comparisons = pd.DataFrame(
        {
            "strategy_a": ["typiclust", "typiclust_adaptive", "typiclust_adaptive"],
            "strategy_b": ["random", "random", "typiclust"],
            "paired_samples": [3, 3, 3],
            "mean_a": [0.61, 0.66, 0.66],
            "mean_b": [0.52, 0.52, 0.61],
            "mean_diff": [0.09, 0.14, 0.05],
            "std_diff": [0.01, 0.02, 0.01],
            "p_value_ttest": [0.04, 0.01, 0.03],
            "p_value_wilcoxon": [0.05, 0.02, 0.04],
        }
    )
    comparisons.to_csv(artifacts_dir / "statistical_comparisons.csv", index=False)

    (artifacts_dir / "accuracy_vs_round.png").write_bytes(b"png")
    (artifacts_dir / "final_round_boxplot.png").write_bytes(b"png")

    main(
        [
            "--artifacts-dir",
            str(artifacts_dir),
            "--report-dir",
            str(report_dir),
            "--pretrain-epochs",
            "500",
            "--gpu-name",
            "RTX 3070",
        ]
    )

    rows = (report_dir / "generated_results_rows.tex").read_text(encoding="utf-8")
    results = (report_dir / "generated_results_text.tex").read_text(encoding="utf-8")
    stats = (report_dir / "generated_stats_text.tex").read_text(encoding="utf-8")
    conclusion = (report_dir / "generated_conclusion_text.tex").read_text(encoding="utf-8")

    assert "TypiClust+Adaptive" in rows
    assert "500" in results
    assert "RTX 3070" in results
    assert "p=" in stats
    assert "TypiClust+Adaptive" in conclusion
    assert (report_dir / "accuracy_vs_round.png").exists()
    assert (report_dir / "final_round_boxplot.png").exists()


def test_fill_report_from_outputs_uses_honest_conclusion_when_random_is_best(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    report_dir = tmp_path / "report"
    artifacts_dir.mkdir()
    report_dir.mkdir()

    summary = pd.DataFrame(
        {
            "strategy": [
                "random",
                "random",
                "typiclust",
                "typiclust",
                "typiclust_adaptive",
                "typiclust_adaptive",
            ],
            "round": [1, 5, 1, 5, 1, 5],
            "mean_accuracy": [0.50, 0.60, 0.54, 0.56, 0.55, 0.57],
            "std_accuracy": [0.01] * 6,
            "se_accuracy": [0.005] * 6,
        }
    )
    summary.to_csv(artifacts_dir / "summary_by_round.csv", index=False)

    comparisons = pd.DataFrame(
        {
            "strategy_a": ["typiclust", "typiclust_adaptive", "typiclust_adaptive"],
            "strategy_b": ["random", "random", "typiclust"],
            "paired_samples": [3, 3, 3],
            "mean_a": [0.56, 0.57, 0.57],
            "mean_b": [0.60, 0.60, 0.56],
            "mean_diff": [-0.04, -0.03, 0.01],
            "std_diff": [0.01, 0.01, 0.01],
            "p_value_ttest": [0.40, 0.35, 0.20],
            "p_value_wilcoxon": [0.50, 0.50, 0.25],
        }
    )
    comparisons.to_csv(artifacts_dir / "statistical_comparisons.csv", index=False)

    (artifacts_dir / "accuracy_vs_round.png").write_bytes(b"png")
    (artifacts_dir / "final_round_boxplot.png").write_bytes(b"png")

    main(
        [
            "--artifacts-dir",
            str(artifacts_dir),
            "--report-dir",
            str(report_dir),
            "--pretrain-epochs",
            "500",
            "--gpu-name",
            "RTX 3070",
        ]
    )

    conclusion = (report_dir / "generated_conclusion_text.tex").read_text(encoding="utf-8")
    stats = (report_dir / "generated_stats_text.tex").read_text(encoding="utf-8")

    assert "Random remained the best final strategy" in conclusion
    assert "strongest overall result" not in conclusion
    assert "directional rather than conclusive" in stats


def test_fill_report_from_outputs_conclusion_reflects_when_adaptive_beats_entropy(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    report_dir = tmp_path / "report"
    artifacts_dir.mkdir()
    report_dir.mkdir()

    summary = pd.DataFrame(
        {
            "strategy": [
                "random",
                "random",
                "entropy",
                "entropy",
                "typiclust",
                "typiclust",
                "typiclust_adaptive",
                "typiclust_adaptive",
            ],
            "round": [1, 5, 1, 5, 1, 5, 1, 5],
            "mean_accuracy": [0.50, 0.61, 0.49, 0.59, 0.53, 0.56, 0.54, 0.60],
            "std_accuracy": [0.01] * 8,
            "se_accuracy": [0.005] * 8,
        }
    )
    summary.to_csv(artifacts_dir / "summary_by_round.csv", index=False)

    comparisons = pd.DataFrame(
        {
            "strategy_a": ["typiclust", "typiclust_adaptive", "typiclust_adaptive"],
            "strategy_b": ["random", "random", "typiclust"],
            "paired_samples": [3, 3, 3],
            "mean_a": [0.56, 0.60, 0.60],
            "mean_b": [0.61, 0.61, 0.56],
            "mean_diff": [-0.05, -0.01, 0.04],
            "std_diff": [0.01, 0.01, 0.01],
            "p_value_ttest": [0.40, 0.30, 0.08],
            "p_value_wilcoxon": [0.50, 0.50, 0.12],
        }
    )
    comparisons.to_csv(artifacts_dir / "statistical_comparisons.csv", index=False)

    (artifacts_dir / "accuracy_vs_round.png").write_bytes(b"png")
    (artifacts_dir / "final_round_boxplot.png").write_bytes(b"png")

    main(
        [
            "--artifacts-dir",
            str(artifacts_dir),
            "--report-dir",
            str(report_dir),
            "--pretrain-epochs",
            "500",
        ]
    )

    conclusion = (report_dir / "generated_conclusion_text.tex").read_text(encoding="utf-8")

    assert "Random remained the best final strategy" in conclusion
    assert "surpassed Entropy" in conclusion
    assert "trailed Random and Entropy" not in conclusion
