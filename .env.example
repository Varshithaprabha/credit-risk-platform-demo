"""
Central configuration. Reads from environment variables (.env) with sane
defaults so the app works out-of-the-box inside Docker.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", BASE_DIR / "data"))
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", BASE_DIR / "data" / "credit_risk.db"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))

# LLM provider for the Talk-to-Data module. Groq's API is OpenAI-compatible,
# so we use the `openai` SDK pointed at Groq's base URL rather than a
# Groq-specific SDK.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

MAX_SQL_RESULT_ROWS = int(os.getenv("MAX_SQL_RESULT_ROWS", "200"))

# Optional password gate for the UI. Leave unset for local/Docker use (no
# friction for an evaluator running `docker-compose up`). Set this as a
# secret on a public deployment (e.g. Hugging Face Spaces) to require a
# password before the app renders anything.
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

RANDOM_STATE = 42

# Core table we build the model + chatbot on top of.
MAIN_TABLE = "application"

# Risk bands derived from predicted default probability.
RISK_BANDS = [
    (0.0, 0.10, "Low"),
    (0.10, 0.30, "Medium"),
    (0.30, 1.01, "High"),
]

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)