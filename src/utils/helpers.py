"""Small shared utility functions."""
from src.utils.config import RISK_BANDS


def score_to_band(prob: float) -> str:
    """Map a predicted default probability (0-1) to a business risk band."""
    for lo, hi, label in RISK_BANDS:
        if lo <= prob < hi:
            return label
    return "High"


def safe_div(a, b):
    return a / b if b else 0.0
