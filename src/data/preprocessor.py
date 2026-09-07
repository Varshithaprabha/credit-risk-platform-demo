"""
Cleaning, encoding, imputation and feature engineering for the Home Credit
application table. Kept deliberately simple/explainable (per the assignment's
"well-explained over complex" guidance) rather than a huge feature-engineering
stack across all 8 tables.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils.logger import get_logger

log = get_logger(__name__)

TARGET = "TARGET"
ID_COL = "SK_ID_CURR"

# Columns that are known data-quality problems in this dataset (Home Credit's
# DAYS_EMPLOYED has a well-known sentinel value of 365243 for "not employed").
DAYS_EMPLOYED_ANOMALY = 365243


def _downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink memory footprint by downcasting to the smallest safe numeric
    dtype (float64->float32, int64->smaller int). Matters a lot on
    memory-constrained deployments (e.g. free-tier cloud hosting) where the
    full dataframe getting copied a few times during the pipeline can add up."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _downcast_dtypes(df)

    # 1. Fix the DAYS_EMPLOYED anomaly (~18% of rows in the real dataset)
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY).astype(int)
        df.loc[df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY, "DAYS_EMPLOYED"] = np.nan

    # 2. Convert DAYS_* (negative, relative to application date) into positive years
    for col in ["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH"]:
        if col in df.columns:
            df[col.replace("DAYS_", "YEARS_")] = (-df[col] / 365.25).round(2)

    # 3. Drop columns that are >60% missing (common threshold, keeps the model lean)
    missing_frac = df.isna().mean()
    high_missing_cols = missing_frac[missing_frac > 0.6].index.tolist()
    high_missing_cols = [c for c in high_missing_cols if c not in (TARGET, ID_COL)]
    if high_missing_cols:
        log.info(f"Dropping {len(high_missing_cols)} columns >60% missing")
        df = df.drop(columns=high_missing_cols)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6

    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + eps)
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + eps)
    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(df.columns):
        df["CREDIT_TERM"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + eps)
    if {"YEARS_EMPLOYED", "YEARS_BIRTH"}.issubset(df.columns):
        df["EMPLOYED_TO_AGE_RATIO"] = df["YEARS_EMPLOYED"] / (df["YEARS_BIRTH"] + eps)
    if "CNT_FAM_MEMBERS" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["INCOME_PER_FAMILY_MEMBER"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + eps)

    return df


def encode_and_impute(df: pd.DataFrame, encoders: dict | None = None):
    """Label-encode categoricals, median/mode-impute numerics.
    Returns (df, encoders) so the same encoders can be reused at inference time.
    """
    df = df.copy()
    encoders = encoders or {}

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in (TARGET, ID_COL)]

    for col in cat_cols:
        df[col] = df[col].fillna("Missing").astype(str)
        if col in encoders:
            le = encoders[col]
            # unseen categories at inference time fall back to a single bucket
            df[col] = df[col].map(lambda v: v if v in le.classes_ else "Missing")
            if "Missing" not in le.classes_:
                le.classes_ = np.append(le.classes_, "Missing")
        else:
            le = LabelEncoder()
            le.fit(df[col])
            encoders[col] = le
        df[col] = le.transform(df[col])

    for col in num_cols:
        median = encoders.get(f"__median__{col}")
        if median is None:
            median = df[col].median()
            encoders[f"__median__{col}"] = median
        df[col] = df[col].fillna(median)

    return df, encoders


def categorize_features(df: pd.DataFrame) -> dict:
    """Business-friendly grouping of raw columns, used in the EDA section of the UI."""
    groups = {
        "Demographics": [c for c in df.columns if c in (
            "CODE_GENDER", "YEARS_BIRTH", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
            "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE", "NAME_HOUSING_TYPE")],
        "Financials": [c for c in df.columns if c in (
            "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
            "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM",
            "INCOME_PER_FAMILY_MEMBER")],
        "Employment": [c for c in df.columns if c in (
            "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "YEARS_EMPLOYED",
            "ORGANIZATION_TYPE", "EMPLOYED_TO_AGE_RATIO")],
        "Credit history / external": [c for c in df.columns if c.startswith("EXT_SOURCE")],
        "Application metadata": [c for c in df.columns if c in (
            "NAME_CONTRACT_TYPE", "WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START")],
    }
    return {k: v for k, v in groups.items() if v}


def full_pipeline(df: pd.DataFrame, encoders: dict | None = None):
    df = clean(df)
    df = engineer_features(df)
    df, encoders = encode_and_impute(df, encoders)
    return df, encoders