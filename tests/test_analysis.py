import pandas as pd

from cw2_ml.reporting.analysis import paired_summary


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
