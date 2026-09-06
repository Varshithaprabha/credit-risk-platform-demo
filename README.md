# AI-Powered Credit Risk Intelligence Platform

A lightweight, end-to-end credit risk platform built on the [Home Credit
Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)
dataset — built for the NeoStats AI Engineer candidate assignment.

## 1. Architecture

```
                     ┌───────────────────────────┐
   Home Credit CSVs  │        SQLite DB          │
   (data/*.csv) ───► │   src/data/loader.py      │
                     └─────────────┬─────────────┘
                                   │
              ┌────────────────────┼─────────────────────┐
              ▼                    ▼                      ▼
     ┌──────────────────┐ ┌────────────────┐   ┌─────────────────────┐
     │ ML Pipeline       │ │ Talk-to-Data   │   │ EDA                 │
     │ preprocessor.py   │ │ nl_to_sql.py   │   │ notebooks/eda.py    │
     │ train.py          │ │ query_runner.py│   │ (pandas/plotly)     │
     │ evaluate.py        │ │ prompt_templates│   └─────────────────────┘
     │ explain.py (SHAP)  │ │ (Groq LLM)   │
     │ rules.py            │ └────────────────┘
     │ predict.py          │
     └─────────┬──────────┘
               │
               ▼
       ┌───────────────────────────────────┐
       │   Streamlit UI (app.py)            │
       │   Overview | EDA | Prediction |    │
       │   Explainability | Rules | Chatbot │
       └───────────────────────────────────┘
```

## 2. Setup & Run

**Prerequisites:** Docker + Docker Compose, and a Groq API key (for the
Talk-to-Data chatbot only — everything else works without it).

```bash
# 1. Download the dataset from Kaggle and place the CSVs in ./data
#    (at minimum: application_train.csv)
#    https://www.kaggle.com/competitions/home-credit-default-risk/data

# 2. Configure environment
cp .env.example .env
# edit .env and add your GROQ_API_KEY (get one free, no card, at https://console.groq.com/keys)

# 3. Run
docker-compose up --build

# 4. Open http://localhost:8501
```

On first load, use the sidebar buttons to (a) build the SQLite database from
the CSVs and (b) train the model. Both are one-click and only need to be done
once (artifacts persist in `./data` and `./models` via the mounted volumes).

**Running locally without Docker:**
```bash
pip install -r requirements.txt
python -m src.data.loader      # builds the SQLite DB
python -m src.ml.train         # trains + saves the model
streamlit run app.py
```

## 3. Model Selection & Class Imbalance Strategy

The dataset has an **~8% default rate** (severe imbalance). Plain accuracy is
misleading here — a model that always predicts "no default" scores ~92%
accuracy while being business-useless. Design choices:

- **Model:** LightGBM (gradient-boosted trees). Chosen over logistic
  regression for its ability to capture non-linear interactions (e.g.
  income × employment length) without heavy manual feature crossing, and
  over deep learning because tabular GBMs consistently outperform neural
  nets on datasets this size/shape while training in minutes on CPU and
  staying explainable via SHAP's exact TreeExplainer.
- **Imbalance handling:** `scale_pos_weight` (re-weights the loss function)
  rather than SMOTE/oversampling. This avoids generating synthetic financial
  profiles that don't correspond to real applicants — an easier thing to
  defend to a credit-risk auditor than "we invented data."
- **Evaluation:** ROC-AUC and PR-AUC (PR-AUC is the more informative one
  under imbalance) are the headline metrics; precision/recall/F1 at a
  chosen threshold are reported for the operating point actually used to
  assign risk bands.
- **Risk bands:** Low (<10% predicted default probability), Medium
  (10–30%), High (≥30%) — thresholds are configurable in `src/utils/config.py`.

## 4. Explainable AI

`src/ml/explain.py` uses SHAP's `TreeExplainer` (exact, not sampled, for
tree models) to produce:
- **Global importance** — which features drive risk across the whole book
- **Local explanation** — for one applicant, the top features pushing their
  score up or down, in plain English ("EXT_SOURCE_2 = 0.21 → increases risk")

## 5. Business Rule Derivation

`src/ml/rules.py` fits a shallow (depth ≤ 4) decision-tree **surrogate**
trained to reproduce the LightGBM model's predicted probabilities (not the
raw labels). The surrogate can't match the full model's accuracy — that's
expected — its job is to translate the black-box score into auditable
if/then policy language a credit committee can actually review, e.g.:

```
IF EXT_SOURCE_2 <= 0.35 AND CREDIT_INCOME_RATIO > 4.2 THEN predicted_default_prob ≈ 0.41 -> High risk
```

## 6. LLM Choice

The assignment allows "any LLM of your choice." This project uses **Groq**
(`llama-3.3-70b-versatile`) rather than the originally-scaffolded Claude,
called via Groq's OpenAI-compatible Chat Completions API
(`https://api.groq.com/openai/v1`). Groq was chosen specifically because it
offers a genuinely free, no-credit-card developer tier (rate-limited rather
than credit-limited), which keeps this project runnable end-to-end by an
evaluator without requiring them to add billing to any account. Its
inference is also notably fast, which keeps the chatbot responsive. The
swap only touches `src/talk_to_data/nl_to_sql.py` and `src/utils/config.py`
— the prompt templates, SQL validation layer, and every other guardrail
described below are provider-agnostic and unchanged. Switching providers
again later only requires changing `GROQ_BASE_URL`/`LLM_MODEL` and the SDK
client init, since any OpenAI-compatible endpoint works as a drop-in.

## 7. Talk-to-Data: Prompt Engineering & Hallucination Control

`src/talk_to_data/` implements NL → SQL → plain-English answer using Groq (via its OpenAI-compatible API).
Guardrails, in order of execution:

1. **Schema-grounded prompt** — only real table/column names (pulled live
   from the SQLite schema) are shown to the model; it cannot see or
   reference anything else.
2. **Few-shot examples** — 5 example Q→SQL pairs teach the exact dialect
   and style expected (see `prompt_templates.py`).
3. **`NO_QUERY` escape hatch** — the model is explicitly told to decline
   rather than guess when a question can't be answered from the schema.
4. **Independent SQL validation** (`query_runner.py`) — every generated
   query is parsed and checked before execution: SELECT-only, single
   statement, only known tables, row-count capped (`MAX_SQL_RESULT_ROWS`).
   The LLM is never trusted to self-enforce these rules.
5. **Grounded answer generation** — the final plain-English answer is
   generated only from the actual returned rows, never from the model's
   general knowledge, which is what prevents fabricated numbers.

**Token optimization:** the schema block sent to the LLM caps columns per
table and only includes a hand-picked glossary of ~15 commonly-referenced
columns rather than the full 200+ row Home Credit column-description file —
keeping each request small and cheap while still grounding it enough to stop
hallucinated column names.

Tested query patterns (≥5, per requirement):
1. Overall default rate
2. Default rate by education / occupation (grouped aggregate)
3. Average income: defaulters vs non-defaulters
4. Top-N occupations by default rate with a minimum sample size (HAVING)
5. Count of applicants meeting a filter condition

## 8. Repository Structure

```
credit_risk_platform/
├── data/                     # CSVs + sqlite db (gitignored, mounted in Docker)
├── documents/                # project_presentation.pdf, eda_charts/
├── notebooks/eda.ipynb, eda.py
├── src/
│   ├── data/loader.py, preprocessor.py
│   ├── ml/train.py, predict.py, evaluate.py, explain.py, rules.py
│   ├── talk_to_data/nl_to_sql.py, query_runner.py, prompt_templates.py
│   └── utils/logger.py, config.py, helpers.py, docker_utils.py
├── sql/schema.sql
├── models/                   # saved model artifacts (gitignored)
├── app.py                    # Streamlit multi-section UI
├── Dockerfile, docker-compose.yml, requirements.txt, .env.example
└── README.md
```

## 9. Known Limitations & Possible Improvements

- Only `application_train.csv` is used for modeling; `bureau.csv`,
  `previous_application.csv`, etc. are loaded into SQLite for the chatbot
  to query but not yet joined into engineered features for the model —
  aggregating them (e.g. count of prior delinquencies) would likely improve
  ROC-AUC further.
- The rule-derivation surrogate trades accuracy for readability by design;
  it should be presented as a policy approximation, not a replacement, for
  the full model's score.
- The Talk-to-Data agent handles single-table (`application`) questions
  well; multi-table joins (e.g. "applicants with 3+ prior loans") work but
  aren't covered by the few-shot examples yet, so results are less reliable there.
- No authentication/rate-limiting on the Streamlit app — fine for a demo,
  not for production banking use.
- Threshold for risk bands (10%/30%) is a reasonable default, not
  calibrated against a specific bank's actual loss-given-default economics.
