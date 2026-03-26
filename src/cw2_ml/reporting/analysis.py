from __future__ import annotations

import math

import pandas as pd
from scipy import stats


def summarize_by_round(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["strategy", "round", "mean_accuracy", "std_accuracy", "se_accuracy"])
    grouped = (
        df.groupby(["strategy", "round"], as_index=False)
        .agg(
            mean_accuracy=("test_accuracy", "mean"),
            std_accuracy=("test_accuracy", "std"),
            n=("test_accuracy", "count"),
        )
        .fillna(0.0)
    )
    grouped["se_accuracy"] = grouped["std_accuracy"] / grouped["n"].map(lambda n: math.sqrt(max(int(n), 1)))
    return grouped.drop(columns=["n"])


def final_round_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    max_round = int(df["round"].max())
    return df[df["round"] == max_round].copy()


def paired_summary(
    df: pd.DataFrame,
    strategy_a: str,
    strategy_b: str,
) -> dict[str, float | str]:
    filtered = df[df["strategy"].isin([strategy_a, strategy_b])].copy()
    if filtered.empty:
        raise ValueError("No rows found for selected strategies.")

    pivot = filtered.pivot_table(
        index=["repeat", "round"],
        columns="strategy",
        values="test_accuracy",
        aggfunc="mean",
    ).dropna()
    if strategy_a not in pivot.columns or strategy_b not in pivot.columns:
        raise ValueError("Could not construct paired comparison with overlapping runs.")

    a = pivot[strategy_a]
    b = pivot[strategy_b]
    diff = a - b
    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0

    if len(diff) >= 2 and not (diff == 0).all():
        t_result = stats.ttest_rel(a, b, alternative="greater", nan_policy="omit")
        p_t = float(1.0 if pd.isna(t_result.pvalue) else t_result.pvalue)
    else:
        p_t = 1.0
    if len(diff) >= 2 and not (diff == 0).all():
        try:
            p_w = float(stats.wilcoxon(diff, alternative="greater").pvalue)
        except ValueError:
            p_w = 1.0
    else:
        p_w = 1.0

    return {
        "strategy_a": strategy_a,
        "strategy_b": strategy_b,
        "paired_samples": int(len(diff)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "p_value_ttest": min(max(p_t, 0.0), 1.0),
        "p_value_wilcoxon": min(max(p_w, 0.0), 1.0),
    }
