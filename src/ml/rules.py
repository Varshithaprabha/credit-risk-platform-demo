"""
Business-readable decision rules derived from the ML model.

Approach: fit a shallow (depth<=4) decision-tree "surrogate" that is trained
to reproduce the LightGBM model's predicted probabilities (not the raw
labels). A shallow tree can't match the full model's accuracy, but its
if/then splits ARE the kind of policy language a credit committee already
uses ("if annuity/income > 0.45 and external score < 0.3 -> High risk"),
which is the point of this module: translating a black-box score into
auditable policy rules, not replacing the model.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, _tree

from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger

log = get_logger(__name__)


def derive_rules(model, X: pd.DataFrame, max_depth: int = 4, top_features: int = 8) -> tuple[DecisionTreeRegressor, list[str]]:
    proba = model.predict_proba(X)[:, 1]

    # restrict the surrogate to the most important raw features so rules stay readable
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_cols = importances.sort_values(ascending=False).head(top_features).index.tolist()

    surrogate = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=500, random_state=42)
    surrogate.fit(X[top_cols], proba)

    rules = _extract_rules(surrogate, top_cols)
    return surrogate, rules


def _extract_rules(tree_model: DecisionTreeRegressor, feature_names: list[str]) -> list[str]:
    tree_ = tree_model.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined"
        for i in tree_.feature
    ]
    rules = []

    def recurse(node, conditions):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = round(tree_.threshold[node], 3)
            recurse(tree_.children_left[node], conditions + [f"{name} <= {threshold}"])
            recurse(tree_.children_right[node], conditions + [f"{name} > {threshold}"])
        else:
            avg_prob = round(float(tree_.value[node][0][0]), 3)
            band = "High" if avg_prob >= 0.30 else "Medium" if avg_prob >= 0.10 else "Low"
            rule = f"IF {' AND '.join(conditions)} THEN predicted_default_prob ≈ {avg_prob} -> {band} risk"
            rules.append(rule)

    recurse(0, [])
    # keep the most decisive rules (highest/lowest predicted prob) first
    rules.sort(key=lambda r: float(r.split("≈ ")[1].split(" ->")[0]), reverse=True)
    return rules


def save_rules(rules: list[str], path: Path = MODELS_DIR / "business_rules.txt") -> None:
    with open(path, "w") as f:
        f.write("\n".join(rules))
    log.info(f"Saved {len(rules)} business rules to {path}")
