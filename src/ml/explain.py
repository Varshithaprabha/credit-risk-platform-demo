"""
Explainable AI layer using SHAP (TreeExplainer, exact for LightGBM — fast,
no sampling approximation needed). Produces:
  - global feature importance (which features drive risk across the book)
  - per-applicant local explanation (why THIS applicant got THIS score)
in a format a non-technical credit analyst can read.
"""
import numpy as np
import pandas as pd
import shap


class RiskExplainer:
    def __init__(self, model, feature_names: list[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def global_importance(self, X: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
        shap_values = self.explainer.shap_values(X)
        vals = shap_values[1] if isinstance(shap_values, list) else shap_values
        mean_abs = np.abs(vals).mean(axis=0)
        out = pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
        return out.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)

    def explain_one(self, x_row: pd.DataFrame, top_n: int = 5) -> list[dict]:
        """x_row: single-row DataFrame with the same columns used at training time."""
        shap_values = self.explainer.shap_values(x_row)
        vals = shap_values[1] if isinstance(shap_values, list) else shap_values
        vals = vals[0]

        contributions = pd.DataFrame({
            "feature": x_row.columns,
            "value": x_row.iloc[0].values,
            "shap_value": vals,
        })
        contributions["direction"] = np.where(
            contributions["shap_value"] > 0, "increases risk", "decreases risk"
        )
        contributions["abs_shap"] = contributions["shap_value"].abs()
        top = contributions.sort_values("abs_shap", ascending=False).head(top_n)

        return [
            {
                "feature": r["feature"],
                "value": r["value"],
                "direction": r["direction"],
                "impact": round(float(r["shap_value"]), 4),
            }
            for _, r in top.iterrows()
        ]

    @staticmethod
    def to_plain_english(explanations: list[dict]) -> str:
        lines = []
        for e in explanations:
            lines.append(f"- {e['feature']} = {e['value']} → {e['direction']} "
                          f"(impact score {e['impact']:+.3f})")
        return "\n".join(lines)
