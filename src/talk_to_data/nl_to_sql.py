"""
The Talk-to-Data agent: natural language question -> validated SQL -> plain
English answer, using Groq as the LLM.

Groq (console.groq.com) is accessed via its OpenAI-compatible Chat
Completions endpoint (https://api.groq.com/openai/v1), so this uses the
standard `openai` Python SDK with its base_url pointed at Groq rather than a
Groq-specific SDK. Groq offers a genuinely free, no-credit-card developer
tier (rate-limited, not credit-limited), which is why it's used here.
Swapping to a different OpenAI-compatible provider later only means changing
GROQ_BASE_URL/LLM_MODEL and the API key in .env — no code change needed.

Hallucination controls used here (see README for the full write-up):
  1. Schema-grounded prompt (only real tables/columns are shown to the model)
  2. Few-shot examples of the exact SQL dialect/style expected
  3. NO_QUERY escape hatch so the model can decline instead of guessing
  4. Independent SQL validation layer (query_runner.py) — the LLM is never
     trusted to enforce its own rules
  5. Answer-generation step is grounded ONLY in the actual query results
     (never the model's general knowledge), reducing fabricated figures
"""
from dataclasses import dataclass

import pandas as pd
from openai import OpenAI

from src.data.loader import get_schema_summary
from src.talk_to_data.prompt_templates import (
    SQL_SYSTEM_PROMPT_V1, ANSWER_SYSTEM_PROMPT_V1, COLUMN_GLOSSARY, EXAMPLE_QA_PAIRS,
)
from src.talk_to_data.query_runner import run_query, SQLValidationError
from src.utils.config import GROQ_API_KEY, GROQ_BASE_URL, LLM_MODEL, MAX_SQL_RESULT_ROWS
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class TalkToDataResult:
    question: str
    sql: str | None
    result_df: pd.DataFrame | None
    answer: str
    error: str | None = None


class TalkToDataAgent:
    def __init__(self, api_key: str = GROQ_API_KEY, model: str = LLM_MODEL,
                 base_url: str = GROQ_BASE_URL):
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def _schema_block(self) -> str:
        schema = get_schema_summary()
        lines = []
        for table, cols in schema.items():
            if table == "columns_description":
                continue
            lines.append(f"{table}({', '.join(cols[:40])})")  # cap columns for token budget
        return "\n".join(lines)

    def _examples_block(self) -> str:
        return "\n".join(f"Q: {q}\nSQL: {s}" for q, s in EXAMPLE_QA_PAIRS)

    def generate_sql(self, question: str) -> str:
        system = SQL_SYSTEM_PROMPT_V1.format(
            schema=self._schema_block(),
            glossary=COLUMN_GLOSSARY,
            max_rows=MAX_SQL_RESULT_ROWS,
        )
        user_msg = f"EXAMPLES:\n{self._examples_block()}\n\nQ: {question}\nSQL:"

        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=300,
            temperature=0,  # deterministic SQL generation
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        return raw.strip("`").replace("sql\n", "").strip()

    def summarize_results(self, question: str, sql: str, df: pd.DataFrame) -> str:
        preview = df.head(20).to_markdown(index=False) if not df.empty else "(no rows returned)"
        user_msg = (
            f"Question: {question}\nSQL run: {sql}\nResult rows ({len(df)} total, "
            f"showing up to 20):\n{preview}"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=300,
            temperature=0.2,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT_V1},
                {"role": "user", "content": user_msg},
            ],
        )
        return resp.choices[0].message.content.strip()

    def ask(self, question: str) -> TalkToDataResult:
        sql = self.generate_sql(question)

        if sql.upper().startswith("NO_QUERY"):
            reason = sql.split(":", 1)[-1].strip()
            return TalkToDataResult(question, None, None, reason, error="no_query")

        try:
            df = run_query(sql)
        except SQLValidationError as e:
            log.warning(f"Rejected SQL: {sql} | reason: {e}")
            return TalkToDataResult(
                question, sql, None,
                "I generated a query that didn't pass safety validation, so I "
                "won't run it. Try rephrasing the question.",
                error=str(e),
            )

        answer = self.summarize_results(question, sql, df)
        return TalkToDataResult(question, sql, df, answer)
