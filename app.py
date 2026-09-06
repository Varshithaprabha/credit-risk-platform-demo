"""
Streamlit UI — the single entry point an evaluator opens after
`docker-compose up`. Five sections in one app, per the assignment's
"multi-section UI covering EDA, risk prediction, explainability, rules,
and the chatbot" requirement.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import build_database, load_main_table, get_schema_summary
from src.data.preprocessor import full_pipeline, TARGET, ID_COL, categorize_features, clean, engineer_features
from src.ml.predict import RiskScorer
from src.ml.explain import RiskExplainer
from src.utils.config import DATABASE_PATH, MODELS_DIR, RAW_DATA_DIR
from src.utils.logger import get_logger

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def _svg(name: str) -> str:
    """Read an SVG asset as text (st.image needs the raw XML string, not a path)."""
    return (ASSETS_DIR / name).read_text()


log = get_logger(__name__)
st.set_page_config(
    page_title="AI Credit Risk Intelligence Platform",
    page_icon="🏦",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    /* tighten default top padding */
    .block-container { padding-top: 2rem; }

    /* metric cards */
    div[data-testid="stMetric"] {
        background: #F0F4F3;
        border: 1px solid #DCE5E3;
        border-radius: 10px;
        padding: 14px 16px;
    }
    div[data-testid="stMetric"] label { color: #4A5A57; }

    /* section headers */
    h1 { font-weight: 700; letter-spacing: -0.5px; }
    h2, h3 { font-weight: 600; }

    /* sidebar */
    section[data-testid="stSidebar"] {
        background: #12302C;
    }
    section[data-testid="stSidebar"] * { color: #EAF2F0 !important; }
    section[data-testid="stSidebar"] .stRadio > label { font-weight: 600; }

    /* buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    /* chat bubbles spacing */
    div[data-testid="stChatMessage"] { margin-bottom: 0.5rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers --
@st.cache_data(show_spinner=False)
def _load_app_df():
    return load_main_table()


def _models_ready() -> bool:
    return (MODELS_DIR / "credit_risk_model.pkl").exists()


def _db_ready() -> bool:
    return Path(DATABASE_PATH).exists()


def _render_score(result):
    band_color = {"Low": "#2E7D32", "Medium": "#E68A00", "High": "#C62828"}[result["risk_band"]]
    prob = result["default_probability"] * 100
    c1, c2 = st.columns([1, 1])
    with c1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": band_color},
                "steps": [
                    {"range": [0, 20], "color": "#E6F4EA"},
                    {"range": [20, 50], "color": "#FDF0DC"},
                    {"range": [50, 100], "color": "#FBE4E4"},
                ],
            },
            title={"text": "Default probability"},
        ))
        gauge.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(gauge, use_container_width=True)
    with c2:
        st.markdown(f"### Risk band: :{'green' if band_color=='#2E7D32' else 'orange' if band_color=='#E68A00' else 'red'}[{result['risk_band']}]")
        st.metric("Default probability", f"{prob:.1f}%")
        st.caption("Risk band thresholds are configurable in `src/ml/predict.py`.")


# ------------------------------------------------------------------ sidebar
_logo_col, _title_col = st.sidebar.columns([1, 3])
with _logo_col:
    st.image(_svg("logo.svg"), width=48)
with _title_col:
    st.markdown(
        "<h3 style='margin-bottom:0; margin-top:2px;'>Credit Risk</h3>"
        "<p style='color:#9FB8B3; margin-top:-4px; font-size:0.8rem;'>AI Intelligence Platform</p>",
        unsafe_allow_html=True,
    )
st.sidebar.markdown("---")

SECTION_ICONS = {
    "Overview": "🏠",
    "EDA": "📊",
    "Risk Prediction": "🎯",
    "Explainability": "🔍",
    "Business Rules": "📋",
    "Talk to Data": "💬",
    "About / Architecture": "🧭",
}
section = st.sidebar.radio(
    "Section",
    list(SECTION_ICONS.keys()),
    format_func=lambda s: f"{SECTION_ICONS[s]}  {s}",
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption("SYSTEM STATUS")
if not _db_ready():
    st.sidebar.warning("⚠️ Database not built")
    with st.sidebar.expander("Build database", expanded=True):
        st.caption("Local run: drop Kaggle CSVs in ./data, then:")
        if st.button("Build from ./data"):
            with st.spinner("Loading CSVs into SQLite..."):
                build_database()
            st.rerun()
        st.caption("Deployed/demo run: upload CSVs directly (not stored in the repo).")
        uploaded = st.file_uploader(
            "Upload Home Credit CSVs", type="csv", accept_multiple_files=True,
        )
        if uploaded and st.button("Build from uploaded files"):
            RAW_DATA_DIR_PATH = Path(RAW_DATA_DIR)
            RAW_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
            for f in uploaded:
                (RAW_DATA_DIR_PATH / f.name).write_bytes(f.getbuffer())
            with st.spinner("Loading uploaded CSVs into SQLite..."):
                build_database()
            st.rerun()
else:
    st.sidebar.success("✅ Database ready")

if not _models_ready():
    st.sidebar.warning("⚠️ Model not trained")
    if st.sidebar.button("Train model now"):
        from src.ml.train import train
        with st.spinner("Training LightGBM model (this can take a few minutes)..."):
            metrics = train()
        st.sidebar.json(metrics)
        st.rerun()
else:
    st.sidebar.success("✅ Model ready")


# ------------------------------------------------------------------ Overview
if section == "Overview":
    st.image(_svg("hero_banner.svg"), use_column_width=True)
    st.markdown(
        "<h1 style='margin-bottom:0;'>AI-Powered Credit Risk Intelligence Platform</h1>"
        "<p style='color:#5A6B67; font-size:1.05rem;'>End-to-end risk scoring, explainability, "
        "and natural-language analytics on the Home Credit Default Risk dataset.</p>",
        unsafe_allow_html=True,
    )

    if _db_ready():
        df = _load_app_df()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Applications", f"{len(df):,}")
        c2.metric("Default rate", f"{df['TARGET'].mean()*100:.1f}%")
        c3.metric("Features", f"{df.shape[1]}")
        c4.metric("Defaults", f"{df['TARGET'].sum():,}")
        st.markdown("")

    st.markdown("#### What's inside")
    cards = [
        ("📊", "EDA", "Data quality, demographics, financials, and key business insights"),
        ("🎯", "Risk Prediction", "LightGBM model scoring default probability + risk band"),
        ("🔍", "Explainability", "SHAP-based, per-applicant reason codes"),
        ("📋", "Business Rules", "Auditable if/then policy rules derived from the model"),
        ("💬", "Talk to Data", "Ask questions in plain English, get SQL-backed answers"),
    ]
    cols = st.columns(5)
    for col, (icon, title, desc) in zip(cols, cards):
        with col:
            st.markdown(
                f"<div style='background:#F0F4F3; border:1px solid #DCE5E3; "
                f"border-radius:10px; padding:16px; height:150px;'>"
                f"<div style='font-size:1.6rem;'>{icon}</div>"
                f"<div style='font-weight:600; margin-top:6px;'>{title}</div>"
                f"<div style='color:#5A6B67; font-size:0.85rem; margin-top:4px;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    if not _db_ready() or not _models_ready():
        st.info("👈 Use the sidebar to build the database and train the model on first run.")


# ------------------------------------------------------------------------ EDA
elif section == "EDA":
    st.title("Exploratory Data Analysis")
    if not _db_ready():
        st.info("Build the database from the sidebar first.")
    else:
        df = _load_app_df()

        st.subheader("Data quality")
        missing = (df.isna().mean().sort_values(ascending=False) * 100).round(1)
        st.bar_chart(missing.head(15))
        st.caption("Top 15 columns by % missing values.")

        st.subheader("Class balance (default vs repaid)")
        fig = px.pie(df, names=df["TARGET"].map({0: "Repaid", 1: "Defaulted"}),
                     title=f"Default rate: {df['TARGET'].mean()*100:.2f}%")
        st.plotly_chart(fig, use_container_width=True)
        st.warning("Severe class imbalance — accuracy alone is a misleading metric here. "
                   "See the Risk Prediction tab for ROC-AUC / PR-AUC instead.")

        st.subheader("Feature categorization")
        engineered = engineer_features(clean(df))
        groups = categorize_features(engineered)
        for group, cols in groups.items():
            with st.expander(f"{group} ({len(cols)} features)"):
                st.write(cols)

        st.subheader("Business insights")
        col1, col2 = st.columns(2)
        with col1:
            if "NAME_EDUCATION_TYPE" in df.columns:
                by_edu = df.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean().sort_values(ascending=False) * 100
                st.plotly_chart(px.bar(by_edu, title="Default rate by education (%)"), use_container_width=True)
        with col2:
            if "DAYS_BIRTH" in df.columns:
                age = (-df["DAYS_BIRTH"] / 365.25)
                st.plotly_chart(
                    px.histogram(x=age, color=df["TARGET"].map({0: "Repaid", 1: "Defaulted"}),
                                 barmode="overlay", title="Age distribution by outcome",
                                 labels={"x": "Age (years)"}),
                    use_container_width=True)

        ext_cols = [c for c in df.columns if c.startswith("EXT_SOURCE")]
        if ext_cols:
            corr = df[ext_cols + ["TARGET"]].corr()["TARGET"].drop("TARGET")
            st.write("**Correlation of external bureau scores with default:**")
            st.dataframe(corr.rename("correlation"))
            st.caption("External bureau scores are consistently the strongest single predictors.")


# ---------------------------------------------------------------- Prediction
elif section == "Risk Prediction":
    st.title("Loan Default Risk Prediction")
    if not _models_ready():
        st.info("Train the model from the sidebar first.")
    else:
        import json
        metrics_path = MODELS_DIR / "metrics.json"
        if metrics_path.exists():
            metrics = json.load(open(metrics_path))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ROC-AUC", metrics["roc_auc"])
            c2.metric("PR-AUC", metrics["pr_auc"])
            c3.metric("Recall @0.5", metrics["recall_at_0.5"])
            c4.metric("Precision @0.5", metrics["precision_at_0.5"])
            st.caption("PR-AUC matters more than ROC-AUC here given the ~8% default rate.")

        st.subheader("Score an applicant")
        df = _load_app_df()
        mode = st.radio("Input mode", ["Pick an existing applicant", "Enter values manually"], horizontal=True)

        scorer = RiskScorer()

        if mode == "Pick an existing applicant":
            app_id = st.selectbox("SK_ID_CURR", df[ID_COL].head(200).tolist())
            row = df[df[ID_COL] == app_id]
            if st.button("Score this applicant"):
                result = scorer.score(row).iloc[0]
                st.session_state["last_scored_row"] = row
                _render_score(result)
        else:
            income = st.number_input("Annual income (AMT_INCOME_TOTAL)", value=150000.0)
            credit = st.number_input("Loan amount (AMT_CREDIT)", value=500000.0)
            annuity = st.number_input("Annuity (AMT_ANNUITY)", value=25000.0)
            if st.button("Score manual applicant"):
                base_row = df.iloc[[0]].copy()
                base_row["AMT_INCOME_TOTAL"] = income
                base_row["AMT_CREDIT"] = credit
                base_row["AMT_ANNUITY"] = annuity
                result = scorer.score(base_row).iloc[0]
                st.session_state["last_scored_row"] = base_row
                _render_score(result)


# -------------------------------------------------------------- Explainability
elif section == "Explainability":
    st.title("Explainable AI (SHAP)")
    if not _models_ready():
        st.info("Train the model first.")
    elif "last_scored_row" not in st.session_state:
        st.info("Score an applicant in the Risk Prediction tab first, then come back here.")
    else:
        model = joblib.load(MODELS_DIR / "credit_risk_model.pkl")
        encoders = joblib.load(MODELS_DIR / "encoders.pkl")
        feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")

        raw_row = st.session_state["last_scored_row"]
        df, _ = full_pipeline(raw_row.drop(columns=[TARGET], errors="ignore"), encoders=encoders)
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]

        explainer = RiskExplainer(model, feature_names)
        explanations = explainer.explain_one(df, top_n=8)

        st.subheader(f"Why this applicant scored the way they did")
        exp_df = pd.DataFrame(explanations)
        fig = px.bar(exp_df, x="impact", y="feature", orientation="h", color="direction",
                     color_discrete_map={"increases risk": "#D64550", "decreases risk": "#4C956C"},
                     title="Top factors driving this prediction (SHAP)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Plain-English explanation:**")
        st.text(RiskExplainer.to_plain_english(explanations))

        st.subheader("Global feature importance (whole portfolio)")
        full_df = _load_app_df()
        sample = full_df.sample(min(1000, len(full_df)), random_state=42)
        sample_processed, _ = full_pipeline(sample.drop(columns=[TARGET], errors="ignore"), encoders=encoders)
        for col in feature_names:
            if col not in sample_processed.columns:
                sample_processed[col] = 0
        sample_processed = sample_processed[feature_names]
        global_imp = explainer.global_importance(sample_processed, top_n=15)
        st.plotly_chart(px.bar(global_imp, x="mean_abs_shap", y="feature", orientation="h"),
                         use_container_width=True)


# -------------------------------------------------------------- Business Rules
elif section == "Business Rules":
    st.title("Business-Readable Decision Rules")
    if not _models_ready():
        st.info("Train the model first.")
    else:
        from src.ml.rules import derive_rules
        model = joblib.load(MODELS_DIR / "credit_risk_model.pkl")
        feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
        encoders = joblib.load(MODELS_DIR / "encoders.pkl")

        df = _load_app_df().sample(min(20000, len(_load_app_df())), random_state=42)
        processed, _ = full_pipeline(df.drop(columns=[TARGET], errors="ignore"), encoders=encoders)
        for col in feature_names:
            if col not in processed.columns:
                processed[col] = 0
        processed = processed[feature_names]

        depth = st.slider("Rule tree depth (more depth = more rules, less readable)", 2, 5, 3)
        if st.button("Derive rules"):
            with st.spinner("Fitting surrogate decision tree..."):
                _, rules = derive_rules(model, processed, max_depth=depth)
            st.success(f"Derived {len(rules)} rules")
            for r in rules:
                st.code(r, language=None)


# ----------------------------------------------------------------- Talk to Data
elif section == "Talk to Data":
    st.markdown(
        "<h1 style='margin-bottom:0;'>💬 Talk to Data</h1>"
        "<p style='color:#5A6B67;'>Ask questions in plain English — answers are generated "
        "only from real SQL query results, never invented.</p>",
        unsafe_allow_html=True,
    )

    if not _db_ready():
        st.info("Build the database first.")
    else:
        from src.talk_to_data.nl_to_sql import TalkToDataAgent
        from src.utils.config import GROQ_API_KEY

        if not GROQ_API_KEY:
            st.error("Set GROQ_API_KEY in your .env file to use this section.")
        else:
            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []

            example_qs = [
                "What is the overall default rate?",
                "How does default rate vary by education level?",
                "What's the average income of defaulters vs non-defaulters?",
                "Show the top 5 occupations by default rate with at least 100 applicants.",
                "How many applicants have more than 2 children?",
            ]

            top_row = st.columns([3, 1])
            with top_row[0]:
                st.caption("Try an example, or just type your own question below.")
            with top_row[1]:
                if st.button("🗑️ Clear chat", use_container_width=True):
                    st.session_state["chat_history"] = []
                    st.rerun()

            chip_cols = st.columns(len(example_qs))
            picked_example = None
            for c, eq in zip(chip_cols, example_qs):
                with c:
                    if st.button(eq, key=f"eq_{eq}", use_container_width=True):
                        picked_example = eq

            # replay chat history as chat bubbles
            for turn in st.session_state["chat_history"]:
                with st.chat_message("user", avatar="🧑‍💼"):
                    st.markdown(turn["question"])
                with st.chat_message("assistant", avatar="🤖"):
                    if turn.get("sql"):
                        with st.expander("Generated SQL"):
                            st.code(turn["sql"], language="sql")
                    if turn.get("result_df") is not None and not turn["result_df"].empty:
                        st.dataframe(turn["result_df"], use_container_width=True)
                    st.markdown(turn["answer"])

            typed_question = st.chat_input("Ask about the credit risk data...")
            question = picked_example or typed_question

            if question:
                with st.chat_message("user", avatar="🧑‍💼"):
                    st.markdown(question)
                agent = TalkToDataAgent()
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Generating SQL and querying..."):
                        result = agent.ask(question)
                    if result.sql:
                        with st.expander("Generated SQL"):
                            st.code(result.sql, language="sql")
                    if result.result_df is not None and not result.result_df.empty:
                        st.dataframe(result.result_df, use_container_width=True)
                    st.markdown(result.answer)

                st.session_state["chat_history"].append({
                    "question": question,
                    "sql": result.sql,
                    "result_df": result.result_df,
                    "answer": result.answer,
                })


# ------------------------------------------------------------ About / Architecture
elif section == "About / Architecture":
    st.markdown(
        "<h1 style='margin-bottom:0;'>🧭 About this Platform</h1>"
        "<p style='color:#5A6B67;'>Architecture, data flow, and design rationale.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Data & request flow")
    flow_steps = [
        ("📁", "Kaggle CSVs", "Home Credit Default Risk tables"),
        ("🗄️", "SQLite", "Loaded + cleaned via src/data/loader.py"),
        ("🤖", "LightGBM", "Trained with scale_pos_weight for class imbalance"),
        ("🔍", "SHAP", "Per-applicant explanations"),
        ("🖥️", "Streamlit UI", "This app — 6 sections, one entry point"),
    ]
    cols = st.columns(len(flow_steps))
    for i, (col, (icon, title, desc)) in enumerate(zip(cols, flow_steps)):
        with col:
            st.markdown(
                f"<div style='text-align:center;'>"
                f"<div style='font-size:1.8rem;'>{icon}</div>"
                f"<div style='font-weight:600; font-size:0.85rem; margin-top:4px;'>{title}</div>"
                f"<div style='color:#5A6B67; font-size:0.75rem;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if i < len(flow_steps) - 1:
                st.markdown(
                    "<div style='text-align:center; color:#9FB8B3; font-size:1.2rem;'>→</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("")
    st.markdown("#### Talk to Data — how it stays safe")
    st.markdown("""
    Natural language never runs directly against the database. The flow is:

    1. **LLM generates SQL** from the question + a schema summary (never raw table dumps, to control token cost)
    2. **Validator checks the SQL** — must be `SELECT`-only, only whitelisted tables/columns, row cap enforced,
       destructive keywords (`DROP`, `DELETE`, `UPDATE`, `ALTER`, ...) hard-blocked before execution
    3. **SQL actually runs** against SQLite — real numbers, not a guess
    4. **LLM summarizes only the returned rows** into a plain-English answer

    If the question can't be answered from the schema, the agent returns a `NO_QUERY` fallback
    instead of inventing columns or numbers.
    """)

    st.markdown("#### Tech stack")
    stack_cols = st.columns(3)
    with stack_cols[0]:
        st.markdown("**ML**\n- LightGBM\n- scikit-learn\n- SHAP")
    with stack_cols[1]:
        st.markdown("**LLM**\n- Groq (OpenAI-compatible API)\n- `openai` Python SDK")
    with stack_cols[2]:
        st.markdown("**App**\n- Streamlit\n- SQLite\n- Docker / Docker Compose")

    st.markdown("#### Known limitations")
    st.markdown("""
    - Talk to Data currently treats each question independently (no multi-turn memory yet)
    - No authentication — fine for a demo/evaluation, not for production use as-is
    - Model was smoke-tested on synthetic data; retrain on the real dataset before trusting metrics
    """)


# ------------------------------------------------------------------------ footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#9FB8B3; font-size:0.8rem;'>"
    "AI-Powered Credit Risk Intelligence Platform · Built for the NeoStats AI Engineer assignment"
    "</p>",
    unsafe_allow_html=True,
)