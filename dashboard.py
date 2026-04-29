import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "eval_results.db")

st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="🧠",
    layout="wide",
)

# ── helpers ──────────────────────────────────────────────────────────────────

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_evaluations() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = get_conn()
    rows = conn.execute("""
        SELECT e.id, e.run_id, e.question, e.faithfulness, e.relevancy,
               e.completeness, e.rouge_l, e.overall_score,
               e.faithfulness_reason, e.relevancy_reason, e.completeness_reason,
               tr.answer, tr.retrieved_context, e.timestamp
        FROM evaluations e
        JOIN test_runs tr ON e.run_id = tr.id
        ORDER BY e.timestamp ASC
    """).fetchall()
    conn.close()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def load_consistency() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM consistency_reports ORDER BY consistency_score ASC"
    ).fetchall()
    conn.close()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def load_generated_questions() -> list:
    if not os.path.exists(DB_PATH):
        return []
    conn = get_conn()
    rows = conn.execute("SELECT question, category FROM generated_questions").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def score_color(score):
    if score is None:
        return "gray"
    if score >= 0.80:
        return "green"
    if score >= 0.60:
        return "orange"
    return "red"


def score_badge(score):
    if score is None:
        return "⚪ N/A"
    if score >= 0.80:
        return f"🟢 {score:.2f}"
    if score >= 0.60:
        return f"🟡 {score:.2f}"
    return f"🔴 {score:.2f}"


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=60)
    st.title("RAG Eval System")
    st.caption(f"Provider: **{os.getenv('LLM_PROVIDER','groq').upper()}**")
    st.caption(f"RAG App: `{os.getenv('RAG_APP_URL','http://localhost:8000')}`")
    st.divider()

    refresh = st.button("🔄 Refresh Now", use_container_width=True)
    auto_refresh = st.toggle("Auto-refresh (30s)", value=True)
    st.divider()

    qs = load_generated_questions()
    if qs:
        st.markdown(f"**Generated Questions ({len(qs)})**")
        for i, q in enumerate(qs):
            st.caption(f"{i+1}. {q['question'][:55]}...")

if auto_refresh:
    st.markdown(
        '<meta http-equiv="refresh" content="30">',
        unsafe_allow_html=True,
    )

# ── load data ─────────────────────────────────────────────────────────────────

df = load_evaluations()
cons_df = load_consistency()

# ── header ────────────────────────────────────────────────────────────────────

st.markdown("## 🧠 RAG Evaluation Dashboard")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
st.divider()

if df.empty:
    st.info("⏳ No evaluation data yet. Start the orchestrator and wait for the first cycle to complete.")
    st.code("python orchestrator.py", language="bash")
    st.stop()

# ── top metrics ───────────────────────────────────────────────────────────────

total_runs   = len(df)
unique_qs    = df["question"].nunique()
avg_overall  = df["overall_score"].mean()
avg_faith    = df["faithfulness"].mean()
avg_relev    = df["relevancy"].mean()
avg_compl    = df["completeness"].mean()
avg_rouge    = df["rouge_l"].mean()
flagged_cnt  = int(cons_df["flagged"].sum()) if not cons_df.empty else 0

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Total Runs",       total_runs)
c2.metric("Questions Tested", unique_qs)
c3.metric("Overall Score",    f"{avg_overall:.2f}", delta=None)
c4.metric("Faithfulness",     f"{avg_faith:.2f}")
c5.metric("Relevancy",        f"{avg_relev:.2f}")
c6.metric("Completeness",     f"{avg_compl:.2f}")
c7.metric("⚠ Inconsistent",  flagged_cnt, delta=None)

st.divider()

# ── score trend chart ─────────────────────────────────────────────────────────

st.markdown("### 📈 Score Trend Across All Runs")

fig = go.Figure()
fig.add_trace(go.Scatter(y=df["overall_score"],    name="Overall",       line=dict(color="#6366f1", width=3)))
fig.add_trace(go.Scatter(y=df["faithfulness"],     name="Faithfulness",  line=dict(color="#22c55e", width=2)))
fig.add_trace(go.Scatter(y=df["relevancy"],        name="Relevancy",     line=dict(color="#f59e0b", width=2)))
fig.add_trace(go.Scatter(y=df["completeness"],     name="Completeness",  line=dict(color="#3b82f6", width=2)))
fig.add_trace(go.Scatter(y=df["rouge_l"],          name="ROUGE-L",       line=dict(color="#ec4899", width=2, dash="dot")))
fig.add_hline(y=0.75, line_dash="dash", line_color="red", annotation_text="Threshold (0.75)")
fig.update_layout(
    height=320,
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    yaxis=dict(range=[0, 1.05]),
    xaxis_title="Run #",
    yaxis_title="Score",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── consistency alerts ────────────────────────────────────────────────────────

if not cons_df.empty:
    flagged = cons_df[cons_df["flagged"] == 1]
    if not flagged.empty:
        st.markdown("### ⚠ Consistency Alerts")
        for _, row in flagged.iterrows():
            with st.expander(f"🔴 {row['question'][:90]}... — Consistency: {row['consistency_score']:.2f}", expanded=False):
                col1, col2, col3 = st.columns(3)
                col1.metric("Consistency Score", f"{row['consistency_score']:.2f}")
                col2.metric("Contradiction Rate", f"{row['contradiction_rate']*100:.0f}%")
                col3.metric("Drift Score", f"{row['drift_score']:.2f}")

                details = row.get("contradiction_details", "[]")
                if isinstance(details, str):
                    try:
                        details = json.loads(details)
                    except Exception:
                        details = []
                if details:
                    st.markdown("**Contradictions detected:**")
                    for d in details:
                        st.error(f"Run {d.get('run_a')} vs Run {d.get('run_b')}: {d.get('detail', '')}")
        st.divider()

# ── per-question comparison ───────────────────────────────────────────────────

st.markdown("### 🔍 Per-Question Answer Comparison (All Runs)")

questions = df["question"].unique()

cons_map = {}
if not cons_df.empty:
    for _, row in cons_df.iterrows():
        cons_map[row["question"]] = row

for q in questions:
    q_df = df[df["question"] == q].reset_index(drop=True)
    cons = cons_map.get(q, {})
    cons_score = cons.get("consistency_score") if cons else None
    is_flagged = bool(cons.get("flagged", False)) if cons else False

    label = f"{'⚠ ' if is_flagged else '✅ '}{q[:85]}..."
    cons_text = f" | Consistency: {cons_score:.2f}" if cons_score is not None else ""

    with st.expander(f"{label}{cons_text} ({len(q_df)} runs)", expanded=is_flagged):

        if cons_score is not None:
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Consistency", f"{cons_score:.2f}")
            cc2.metric("Contradiction Rate", f"{cons.get('contradiction_rate', 0)*100:.0f}%")
            cc3.metric("Drift", f"{cons.get('drift_score', 0):.2f}")
            cc4.metric("Total Runs", cons.get("total_runs", len(q_df)))
            st.markdown("---")

        for i, row in q_df.iterrows():
            run_label = f"**Run {i+1}**"
            r1, r2, r3, r4, r5, r6 = st.columns([1, 4, 1, 1, 1, 1])

            r1.markdown(run_label)
            r2.caption(row.get("answer", "")[:200])
            r3.markdown(score_badge(row.get("faithfulness")), help=row.get("faithfulness_reason", ""))
            r4.markdown(score_badge(row.get("relevancy")),    help=row.get("relevancy_reason", ""))
            r5.markdown(score_badge(row.get("completeness")), help=row.get("completeness_reason", ""))

            overall = row.get("overall_score")
            color = score_color(overall)
            r6.markdown(
                f'<div style="background:{"#22c55e" if color=="green" else "#f59e0b" if color=="orange" else "#ef4444"};'
                f'color:white;padding:4px 8px;border-radius:6px;text-align:center;font-weight:700;">'
                f'{overall:.2f if overall is not None else "N/A"}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div style='display:flex;gap:24px;margin-top:4px;font-size:12px;color:#64748b;'>"
            "<span>Columns: Run | Answer | 🟢Faithfulness | 🟡Relevancy | 🔵Completeness | Overall</span>"
            "</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ── radar chart ───────────────────────────────────────────────────────────────

st.markdown("### 🎯 Average Score Breakdown")

fig2 = go.Figure(go.Scatterpolar(
    r=[avg_faith, avg_relev, avg_compl, avg_rouge, avg_overall],
    theta=["Faithfulness", "Relevancy", "Completeness", "ROUGE-L", "Overall"],
    fill="toself",
    line_color="#6366f1",
    fillcolor="rgba(99,102,241,0.2)",
))
fig2.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    height=350,
    margin=dict(l=40, r=40, t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig2, use_container_width=True)

# ── raw data table ────────────────────────────────────────────────────────────

with st.expander("📋 Raw Evaluation Data"):
    display_cols = ["run_id", "question", "faithfulness", "relevancy",
                    "completeness", "rouge_l", "overall_score", "timestamp"]
    st.dataframe(
        df[display_cols].rename(columns={
            "run_id": "Run", "question": "Question",
            "faithfulness": "Faith.", "relevancy": "Relev.",
            "completeness": "Compl.", "rouge_l": "ROUGE-L",
            "overall_score": "Overall", "timestamp": "Time"
        }),
        use_container_width=True,
        hide_index=True,
    )

st.caption("🔄 Dashboard auto-refreshes every 30 seconds | Built with Streamlit + Groq/Azure")
