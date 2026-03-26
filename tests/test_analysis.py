import warnings

import pandas as pd

from cw2_ml.reporting.analysis import paired_summary, summarize_by_round


def test_summarize_by_round_keeps_repeat_statistics() -> None:
    df = pd.DataFrame(
        {
            "strategy": ["typiclust", "typiclust", "typiclust_adaptive", "typiclust_adaptive"],
            "repeat": [0, 1, 0, 1],
            "round": [5, 5, 5, 5],
            "test_accuracy": [0.61, 0.65, 0.68, 0.70],
            "labeled_count": [50, 50, 50, 50],
        }
    )

    summary = summarize_by_round(df)

    assert set(summary.columns) >= {"mean_accuracy", "std_accuracy", "se_accuracy"}


def test_paired_summary_returns_p_value() -> None:
    df = pd.DataFrame(
        {
            "strategy": ["typiclust", "typiclust", "random", "random"],
            "repeat": [0, 1, 0, 1],
            "round": [1, 1, 1, 1],
            "test_accuracy": [0.7, 0.72, 0.65, 0.66],
        }
    )
    result = paired_summary(df, strategy_a="typiclust", strategy_b="random")
    assert 0.0 <= result["p_value_ttest"] <= 1.0


def test_paired_summary_avoids_runtime_warnings_for_single_pair() -> None:
    df = pd.DataFrame(
        {
            "strategy": ["typiclust", "random"],
            "repeat": [0, 0],
            "round": [5, 5],
            "test_accuracy": [0.64, 0.60],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = paired_summary(df, strategy_a="typiclust", strategy_b="random")

    assert result["paired_samples"] == 1
    assert not caught
