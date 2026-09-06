"""
Evaluation focused on metrics that are meaningful under class imbalance.
Plain accuracy is intentionally NOT the headline metric — with an 8% default
rate, a model that predicts "no default" for everyone scores ~92% accuracy
while being useless.
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    f1_score, precision_score, recall_score, confusion_matrix,
)


def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    f1 = f1_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    return {
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "f1_at_0.5": round(float(f1), 4),
        "precision_at_0.5": round(float(precision), 4),
        "recall_at_0.5": round(float(recall), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": threshold,
        "n_test": int(len(y_test)),
        "default_rate_test": round(float(y_test.mean()), 4),
    }


def best_threshold_for_recall(model, X_test, y_test, target_recall: float = 0.6) -> float:
    """Credit risk platforms usually favor recall (catch more true defaulters)
    over precision, since a missed defaulter (false negative) is costlier than
    a false alarm. This finds the threshold hitting a target recall."""
    proba = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
    idx = np.argmin(np.abs(recalls[:-1] - target_recall))
    return float(thresholds[idx])
