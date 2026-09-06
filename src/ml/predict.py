"""Inference: takes raw application rows and returns a default probability
and risk band per row, using the saved model + encoders from train.py."""
from pathlib import Path

import joblib
import pandas as pd

from src.data.preprocessor import full_pipeline, TARGET, ID_COL
from src.utils.config import MODELS_DIR
from src.utils.helpers import score_to_band


class RiskScorer:
    def __init__(self, model_dir: Path = MODELS_DIR):
        model_dir = Path(model_dir)
        self.model = joblib.load(model_dir / "credit_risk_model.pkl")
        self.encoders = joblib.load(model_dir / "encoders.pkl")
        self.feature_names = joblib.load(model_dir / "feature_names.pkl")

    def score(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        ids = raw_df[ID_COL] if ID_COL in raw_df.columns else pd.Series(range(len(raw_df)))
        df, _ = full_pipeline(raw_df.drop(columns=[TARGET], errors="ignore"), encoders=self.encoders)

        # align columns to training-time feature set
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names]

        proba = self.model.predict_proba(df)[:, 1]
        bands = [score_to_band(p) for p in proba]

        return pd.DataFrame({
            ID_COL: ids.values,
            "default_probability": proba.round(4),
            "risk_band": bands,
        })

    def score_one(self, record: dict) -> dict:
        df = pd.DataFrame([record])
        result = self.score(df).iloc[0]
        return {
            "default_probability": float(result["default_probability"]),
            "risk_band": result["risk_band"],
        }
