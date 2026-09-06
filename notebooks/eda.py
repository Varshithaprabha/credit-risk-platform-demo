"""
Exploratory Data Analysis for the Home Credit Default Risk dataset.
This is the .py export of notebooks/eda.ipynb (kept in sync — see README).

Run with: python notebooks/eda.py
Produces printed summaries + saves charts to documents/eda_charts/
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.loader import load_main_table
from src.data.preprocessor import clean, engineer_features, categorize_features

OUT_DIR = Path(__file__).resolve().parent.parent / "documents" / "eda_charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")


def main():
    print("=" * 70)
    print("HOME CREDIT DEFAULT RISK — EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    df = load_main_table()
    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # ---- 1. Data quality ----
    missing = df.isna().mean().sort_values(ascending=False)
    print("\nTop 10 columns by missing %:")
    print((missing.head(10) * 100).round(1).astype(str) + "%")

    dupes = df.duplicated(subset=["SK_ID_CURR"]).sum()
    print(f"\nDuplicate application IDs: {dupes}")

    # ---- 2. Class balance (THE key insight for this dataset) ----
    default_rate = df["TARGET"].mean() * 100
    print(f"\nOverall default rate: {default_rate:.2f}%  "
          f"(={df['TARGET'].sum():,} defaults out of {len(df):,} applications)")
    print("-> Severe class imbalance. A naive always-predict-0 model would score "
          f"{100-default_rate:.1f}% accuracy while being useless. Use ROC-AUC/PR-AUC, "
          "not accuracy, and weight the minority class during training.")

    fig, ax = plt.subplots(figsize=(5, 4))
    df["TARGET"].value_counts().plot(kind="bar", ax=ax, color=["#4C956C", "#D64550"])
    ax.set_xticklabels(["Repaid (0)", "Defaulted (1)"], rotation=0)
    ax.set_title("Class Balance: Loan Default (TARGET)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_class_balance.png", dpi=120)
    plt.close(fig)

    # ---- 3. Feature categorization ----
    cleaned = clean(df)
    engineered = engineer_features(cleaned)
    groups = categorize_features(engineered)
    print("\nFeature categorization:")
    for group, cols in groups.items():
        print(f"  {group}: {len(cols)} features")

    # ---- 4. Business insight: default rate by education ----
    if "NAME_EDUCATION_TYPE" in df.columns:
        by_edu = df.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean().sort_values(ascending=False) * 100
        print("\nDefault rate by education level:")
        print(by_edu.round(2))
        fig, ax = plt.subplots(figsize=(7, 4))
        by_edu.plot(kind="barh", ax=ax, color="#3E7CB1")
        ax.set_xlabel("Default rate (%)")
        ax.set_title("Default Rate by Education Level")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "02_default_by_education.png", dpi=120)
        plt.close(fig)

    # ---- 5. Business insight: credit/income ratio vs default ----
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        ratio = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, 1)
        by_default = ratio.groupby(df["TARGET"]).median()
        print(f"\nMedian credit/income ratio — repaid: {by_default[0]:.2f}, "
              f"defaulted: {by_default[1]:.2f}")

    # ---- 6. Business insight: age vs default ----
    if "DAYS_BIRTH" in df.columns:
        age_years = -df["DAYS_BIRTH"] / 365.25
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.kdeplot(age_years[df["TARGET"] == 0], label="Repaid", ax=ax, fill=True)
        sns.kdeplot(age_years[df["TARGET"] == 1], label="Defaulted", ax=ax, fill=True)
        ax.set_xlabel("Applicant age (years)")
        ax.set_title("Age Distribution by Repayment Outcome")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "03_age_vs_default.png", dpi=120)
        plt.close(fig)
        print(f"\nMedian age — repaid: {age_years[df['TARGET']==0].median():.1f}, "
              f"defaulted: {age_years[df['TARGET']==1].median():.1f}  "
              "(younger applicants trend riskier)")

    # ---- 7. Business insight: external source scores ----
    ext_cols = [c for c in df.columns if c.startswith("EXT_SOURCE")]
    if ext_cols:
        corr = df[ext_cols + ["TARGET"]].corr()["TARGET"].drop("TARGET")
        print(f"\nCorrelation of external bureau scores with default:\n{corr.round(3)}")
        print("-> External bureau scores are consistently the strongest single "
              "predictors of default (negative correlation = higher score, lower risk).")

    print(f"\nCharts saved to: {OUT_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()
