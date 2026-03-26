from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cw2_ml.utils.io import ensure_dir


DISPLAY_NAMES = {
    "random": "Random",
    "entropy": "Entropy",
    "typiclust": "TypiClust",
    "typiclust_adaptive": "TypiClust+Adaptive",
}


def _with_display_names(df: pd.DataFrame) -> pd.DataFrame:
    labeled = df.copy()
    labeled["strategy"] = labeled["strategy"].map(lambda name: DISPLAY_NAMES.get(name, name))
    return labeled


def plot_accuracy_by_round(df: pd.DataFrame, output_path: str | Path) -> None:
    if df.empty:
        return
    out = Path(output_path)
    ensure_dir(out.parent)
    plot_df = _with_display_names(df)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=plot_df,
        x="round",
        y="test_accuracy",
        hue="strategy",
        marker="o",
        estimator="mean",
        errorbar="se",
        ax=ax,
    )
    ax.set_title("Test Accuracy Across Active Learning Rounds")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("AL Round")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_final_round_boxplot(df: pd.DataFrame, output_path: str | Path) -> None:
    if df.empty:
        return
    out = Path(output_path)
    ensure_dir(out.parent)

    max_round = int(df["round"].max())
    final_df = _with_display_names(df[df["round"] == max_round].copy())
    if final_df.empty:
        return

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    strategy_order = final_df["strategy"].drop_duplicates().tolist()
    grouped = [
        final_df.loc[final_df["strategy"] == strategy, "test_accuracy"].to_numpy()
        for strategy in strategy_order
    ]
    positions = np.arange(1, len(strategy_order) + 1)
    ax.boxplot(grouped, positions=positions, widths=0.55, patch_artist=False)
    for position, values in zip(positions, grouped):
        jitter = np.linspace(-0.06, 0.06, num=len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(
            position + jitter,
            values,
            color="black",
            alpha=0.6,
            s=18,
            zorder=3,
        )
    ax.set_xticks(positions, strategy_order)
    ax.set_title(f"Final-Round Accuracy Distribution (Round {max_round})")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Strategy")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
