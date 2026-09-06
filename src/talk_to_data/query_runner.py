"""
Executes LLM-generated SQL against the SQLite DB — but only after it passes
validation. This is the hallucination/safety guardrail layer: even if the LLM
ever produced something destructive or off-schema, it physically cannot run.
"""
import re
import sqlite3

import pandas as pd
import sqlparse

from src.data.loader import get_schema_summary
from src.utils.config import DATABASE_PATH, MAX_SQL_RESULT_ROWS
from src.utils.logger import get_logger

log = get_logger(__name__)

BLOCKED_KEYWORDS = {
    "insert", "update", "delete", "drop", "alter", "attach", "detach",
    "pragma", "create", "replace", "vacuum", "reindex", "grant",
}


class SQLValidationError(Exception):
    pass


def validate_sql(sql: str, db_path=DATABASE_PATH) -> str:
    """Raises SQLValidationError if the query is unsafe or off-schema.
    Returns the (possibly LIMIT-clamped) query if it's safe to run."""
    sql_clean = sql.strip().rstrip(";")

    if not sql_clean:
        raise SQLValidationError("Empty query.")

    parsed = sqlparse.parse(sql_clean)
    if len(parsed) != 1:
        raise SQLValidationError("Only a single statement is allowed.")

    statement = parsed[0]
    stmt_type = statement.get_type()
    if stmt_type != "SELECT":
        raise SQLValidationError(f"Only SELECT statements are allowed (got {stmt_type}).")

    lowered = sql_clean.lower()
    for kw in BLOCKED_KEYWORDS:
        if re.search(rf"\b{kw}\b", lowered):
            raise SQLValidationError(f"Query contains a disallowed keyword: {kw}")

    # verify every referenced table exists in the known schema (blocks
    # hallucinated table names outright)
    schema = get_schema_summary(db_path)
    known_tables = set(schema.keys())
    referenced = set(re.findall(r"(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", lowered))
    unknown = referenced - known_tables
    if unknown:
        raise SQLValidationError(f"Unknown table(s) referenced: {', '.join(unknown)}")

    # enforce a row cap even if the LLM forgot / omitted LIMIT
    if "limit" not in lowered and not _looks_like_aggregate(lowered):
        sql_clean += f" LIMIT {MAX_SQL_RESULT_ROWS}"

    return sql_clean


def _looks_like_aggregate(lowered_sql: str) -> bool:
    agg_fns = ("count(", "avg(", "sum(", "min(", "max(")
    has_group_by = "group by" in lowered_sql
    has_agg = any(fn in lowered_sql for fn in agg_fns)
    return has_agg and not has_group_by


def run_query(sql: str, db_path=DATABASE_PATH) -> pd.DataFrame:
    safe_sql = validate_sql(sql, db_path)
    log.info(f"Executing validated SQL: {safe_sql}")
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(safe_sql, conn)
    finally:
        conn.close()
    return df
