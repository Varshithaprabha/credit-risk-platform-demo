"""
Trains a LightGBM classifier to predict loan default (TARGET=1).

Class imbalance strategy: the Home Credit dataset is ~91.9% non-default /
8.1% default. Rather than oversampling (SMOTE) — which can create unrealistic
synthetic financial profiles — we use LightGBM's `scale_pos_weight`, which
re-weights the minority class in the loss function without touching the data
distribution. This is faster, avoids overfitting to synthetic samples, and is
easy to justify to a credit-risk auditor (a documented, transparent weighting
scheme beats undocumented resampling).

Evaluation is threshold-agnostic first (ROC-AUC, PR-AUC — PR-AUC matters more
than ROC-AUC here because of the imbalance) and threshold-specific second
(precision/recall/F1 at the operating point used for risk bands).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from src.data.loader import load_main_table
from src.data.preprocessor import full_pipeline, TARGET, ID_COL
from src.ml.evaluate import evaluate_model
from src.utils.config import MODELS_DIR, RANDOM_STATE
from src.utils.logger import get_logger

log = get_logger(__name__)


def train(save_dir: Path = MODELS_DIR) -> dict:
    log.info("Loading main application table from SQLite...")
    raw = load_main_table()

    log.info("Running preprocessing pipeline (clean -> engineer -> encode/impute)...")
    df, encoders = full_pipeline(raw)

    y = df[TARGET]
    X = df.drop(columns=[TARGET, ID_COL], errors="ignore")
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    log.info(f"Class balance -> negatives={neg}, positives={pos}, "
              f"scale_pos_weight={scale_pos_weight:.2f}")

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    log.info("Training LightGBM...")
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    log.info(f"Test metrics: {json.dumps(metrics, indent=2)}")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_dir / "credit_risk_model.pkl")
    joblib.dump(encoders, save_dir / "encoders.pkl")
    joblib.dump(feature_names, save_dir / "feature_names.pkl")
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"Saved model artifacts to {save_dir}")
    return metrics


if __name__ == "__main__":
    train()
