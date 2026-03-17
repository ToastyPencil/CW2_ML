from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cw2_ml.utils.io import ensure_dir


def plot_accuracy_by_round(df: pd.DataFrame, output_path: str | Path) -> None:
    if df.empty:
        return
    out = Path(output_path)
    ensure_dir(out.parent)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=df,
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
    final_df = df[df["round"] == max_round].copy()
    if final_df.empty:
        return

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=final_df, x="strategy", y="test_accuracy", ax=ax)
    sns.stripplot(data=final_df, x="strategy", y="test_accuracy", color="black", alpha=0.6, ax=ax)
    ax.set_title(f"Final-Round Accuracy Distribution (Round {max_round})")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Strategy")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
