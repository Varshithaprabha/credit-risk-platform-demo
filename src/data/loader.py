"""
Loads the raw Home Credit Default Risk CSVs (as downloaded from Kaggle:
https://www.kaggle.com/competitions/home-credit-default-risk/data) into a
local SQLite database. SQLite is used (instead of a heavier DB) because it
needs zero setup, ships as a single file, and is what the Talk-to-Data /
NL-to-SQL module queries against.

CSVs are read in chunks (not all at once) so peak memory stays low — this
matters a lot on memory-constrained deployments (e.g. Streamlit Community
Cloud's free tier, ~1GB RAM total) where loading a 150MB+ CSV in one shot
can crash the process.

Expected files in RAW_DATA_DIR (only application_train.csv is mandatory —
everything else is optional and enriches the EDA / chatbot if present):
    application_train.csv
    application_test.csv
    bureau.csv
    bureau_balance.csv
    previous_application.csv
    POS_CASH_balance.csv
    credit_card_balance.csv
    installments_payments.csv
    HomeCredit_columns_description.csv
"""
import sqlite3
from pathlib import Path

import pandas as pd

from src.utils.config import RAW_DATA_DIR, DATABASE_PATH, MAIN_TABLE
from src.utils.logger import get_logger

log = get_logger(__name__)

CHUNK_SIZE = 50_000  # rows per chunk — keeps peak memory low regardless of file size

# filename -> sql table name
TABLE_MAP = {
    "application_train.csv": "application",
    "application_test.csv": "application_test",
    "bureau.csv": "bureau",
    "bureau_balance.csv": "bureau_balance",
    "previous_application.csv": "previous_application",
    "POS_CASH_balance.csv": "pos_cash_balance",
    "credit_card_balance.csv": "credit_card_balance",
    "installments_payments.csv": "installments_payments",
    "HomeCredit_columns_description.csv": "columns_description",
}


def _detect_encoding(fpath: Path, sample_bytes: int = 8192) -> str:
    """Cheaply sniff whether a file is UTF-8 or Latin-1/cp1252 (some Home
    Credit files, notably HomeCredit_columns_description.csv, aren't UTF-8)
    without reading the whole file into memory."""
    with open(fpath, "rb") as f:
        raw = f.read(sample_bytes)
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin1"


def build_database(raw_dir: Path = RAW_DATA_DIR, db_path: Path = DATABASE_PATH,
                    sample_frac: float | None = None) -> None:
    """Read every available CSV and write it as a table in the SQLite DB.

    sample_frac: optionally subsample rows (0-1) — useful for a fast local
    demo run before committing to training on the full ~300k-row dataset.
    Reads each CSV in CHUNK_SIZE-row chunks to keep peak memory low.
    """
    raw_dir = Path(raw_dir)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    found_any = False
    for filename, table in TABLE_MAP.items():
        fpath = raw_dir / filename
        if not fpath.exists():
            log.warning(f"Skipping missing file: {filename}")
            continue

        log.info(f"Loading {filename} -> table `{table}` (chunked)")
        encoding = _detect_encoding(fpath)
        first_chunk = True
        rows_written = 0
        for chunk in pd.read_csv(fpath, encoding=encoding, chunksize=CHUNK_SIZE):
            if sample_frac and table in (MAIN_TABLE, "application_test"):
                chunk = chunk.sample(frac=sample_frac, random_state=42)
            chunk.to_sql(table, conn, if_exists="replace" if first_chunk else "append", index=False)
            rows_written += len(chunk)
            first_chunk = False
            del chunk  # release chunk memory before reading the next one

        log.info(f"  -> {rows_written:,} rows written to `{table}`")
        found_any = True

    if not found_any:
        conn.close()
        raise FileNotFoundError(
            f"No Home Credit CSVs found in {raw_dir}. Download the dataset from "
            "https://www.kaggle.com/competitions/home-credit-default-risk/data "
            "and place application_train.csv (at minimum) there."
        )

    _create_indexes(conn)
    conn.close()
    log.info(f"SQLite database ready at {db_path}")


def _create_indexes(conn: sqlite3.Connection) -> None:
    """Index join keys so the talk-to-data agent's SQL runs fast."""
    stmts = [
        "CREATE INDEX IF NOT EXISTS idx_app_id ON application(SK_ID_CURR)",
        "CREATE INDEX IF NOT EXISTS idx_bureau_id ON bureau(SK_ID_CURR)",
        "CREATE INDEX IF NOT EXISTS idx_prev_id ON previous_application(SK_ID_CURR)",
        "CREATE INDEX IF NOT EXISTS idx_pos_id ON pos_cash_balance(SK_ID_CURR)",
        "CREATE INDEX IF NOT EXISTS idx_cc_id ON credit_card_balance(SK_ID_CURR)",
        "CREATE INDEX IF NOT EXISTS idx_inst_id ON installments_payments(SK_ID_CURR)",
    ]
    for s in stmts:
        try:
            conn.execute(s)
        except sqlite3.OperationalError:
            pass  # table doesn't exist in this run — fine, it's optional
    conn.commit()


def load_main_table(db_path: Path = DATABASE_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {MAIN_TABLE}", conn)
    conn.close()
    return df


def get_schema_summary(db_path: Path = DATABASE_PATH) -> dict:
    """Returns {table_name: [column names]} — used to ground the NL->SQL prompt."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    schema = {}
    for t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        schema[t] = [row[1] for row in cur.fetchall()]
    conn.close()
    return schema


if __name__ == "__main__":
    build_database()