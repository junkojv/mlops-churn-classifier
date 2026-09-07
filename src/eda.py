"""
Exploratory Data Analysis (EDA) — Telco Customer Churn
Generates and saves key visualizations to reports/eda/
Usage: python src/eda.py --config configs/config.yaml
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils import load_config

sns.set_theme(style="whitegrid", palette="muted")
OUTPUT_DIR = "reports/eda"


def run_eda(config_path: str = "configs/config.yaml") -> None:
    config = load_config(config_path)
    csv_path = config["data"]["csv_path"]
    target_col = config["data"]["target"]
    num_cols = config["features"]["numeric"]
    cat_cols = config["features"]["categorical"]

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Clean TotalCharges (Telco dataset specificity)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Basic stats ──────────────────────────────────────────────────────
    print(f"\nDataset shape : {df.shape}")
    print(f"Target column : {target_col}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nClass distribution:\n{df[target_col].value_counts()}")

    # ── 2. Churn distribution (bar chart) ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[target_col].value_counts()
    ax.bar(counts.index.astype(str), counts.values,
           color=["#4CAF50", "#F44336"], edgecolor="white")
    ax.set_title("Churn Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 20, f"{v}\n({v/len(df)*100:.1f}%)",
                ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/churn_distribution.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR}/churn_distribution.png")

    # ── 3. Numeric features — distributions by churn ────────────────────────
    fig, axes = plt.subplots(1, len(num_cols), figsize=(5 * len(num_cols), 4))
    if len(num_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, num_cols):
        for label, grp in df.groupby(target_col):
            ax.hist(grp[col].dropna(), bins=30, alpha=0.6, label=str(label))
        ax.set_title(col, fontsize=12)
        ax.set_xlabel(col)
        ax.legend()
    fig.suptitle("Numeric Features Distribution by Churn", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/numeric_distributions.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR}/numeric_distributions.png")

    # ── 4. Correlation heatmap (numeric) ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, ax=ax, linewidths=0.5)
    ax.set_title("Numeric Features — Correlation Heatmap",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR}/correlation_heatmap.png")

    # ── 5. Churn rate by key categorical features ────────────────────────────
    key_cats = [c for c in ["Contract", "InternetService", "PaymentMethod"] if c in cat_cols]
    if key_cats:
        fig, axes = plt.subplots(1, len(key_cats), figsize=(6 * len(key_cats), 5))
        if len(key_cats) == 1:
            axes = [axes]
        for ax, col in zip(axes, key_cats):
            churn_rate = df.groupby(col)[target_col].apply(
                lambda s: (s.isin(["Yes", 1, True])).mean() * 100
            ).sort_values(ascending=False)
            sns.barplot(x=churn_rate.index, y=churn_rate.values,
                        hue=churn_rate.index, legend=False,
                        ax=ax, palette="Reds_r")
            ax.set_title(f"Churn Rate by {col}", fontsize=12)
            ax.set_ylabel("Churn Rate (%)")
            ax.set_xlabel(col)
            ax.tick_params(axis="x", rotation=15)
            for i, v in enumerate(churn_rate.values):
                ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
        fig.suptitle("Churn Rate by Key Categorical Features",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(f"{OUTPUT_DIR}/churn_rate_by_category.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {OUTPUT_DIR}/churn_rate_by_category.png")

    print(f"\nAll EDA plots saved to '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_eda(args.config)
