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
               e.factual_anchor_score, e.factual_hallucinated,
               e.golden_rouge_l, e.contradicts_golden, e.contradiction_detail,
               tr.answer, tr.retrieved_context, e.timestamp
        FROM evaluations e
        JOIN test_runs tr ON e.run_id = tr.id
        ORDER BY e.timestamp ASC
    """).fetchall()
    conn.close()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def load_golden_answers() -> dict:
    if not os.path.exists(DB_PATH):
        return {}
    conn = get_conn()
    rows = conn.execute("SELECT question, golden_answer FROM golden_answers").fetchall()
    conn.close()
    return {r["question"]: r["golden_answer"] for r in rows}


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
golden_map = load_golden_answers()

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
avg_factual  = df["factual_anchor_score"].mean() if "factual_anchor_score" in df.columns and df["factual_anchor_score"].notna().any() else None
avg_golden_r = df["golden_rouge_l"].mean() if "golden_rouge_l" in df.columns and df["golden_rouge_l"].notna().any() else None
flagged_cnt  = int(cons_df["flagged"].sum()) if not cons_df.empty else 0
contradicts_cnt = int(df["contradicts_golden"].sum()) if "contradicts_golden" in df.columns else 0

st.markdown("#### Layer 1 & 2 — Grounded (Zero LLM)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Factual Anchor Score", f"{avg_factual:.2f}" if avg_factual is not None else "N/A",
          help="Pure code: facts in answer vs source context")
c2.metric("Golden ROUGE-L",       f"{avg_golden_r:.2f}" if avg_golden_r is not None else "N/A",
          help="Math: text overlap with golden reference answer")
c3.metric("Contradicts Golden",   contradicts_cnt,
          help="Runs where answer contradicts the ground truth")
c4.metric("⚠ Inconsistent Qs",   flagged_cnt)

st.markdown("#### Layer 3 — LLM Judge (Grounded by Golden Answer)")
c5, c6, c7, c8, c9 = st.columns(5)
c5.metric("Overall Score",  f"{avg_overall:.2f}")
c6.metric("Faithfulness",   f"{avg_faith:.2f}")
c7.metric("Relevancy",      f"{avg_relev:.2f}")
c8.metric("Completeness",   f"{avg_compl:.2f}")
c9.metric("Total Runs",     total_runs)

st.divider()

# ── score trend chart ─────────────────────────────────────────────────────────

st.markdown("### 📈 Score Trend Across All Runs")

fig = go.Figure()
fig.add_trace(go.Scatter(y=df["overall_score"],    name="Overall",             line=dict(color="#6366f1", width=3)))
if "factual_anchor_score" in df.columns:
    fig.add_trace(go.Scatter(y=df["factual_anchor_score"], name="Factual Anchors (L1)", line=dict(color="#dc2626", width=2)))
if "golden_rouge_l" in df.columns:
    fig.add_trace(go.Scatter(y=df["golden_rouge_l"],       name="Golden ROUGE-L (L2)", line=dict(color="#7c3aed", width=2)))
fig.add_trace(go.Scatter(y=df["faithfulness"],     name="Faithfulness (L3)",   line=dict(color="#22c55e", width=2)))
fig.add_trace(go.Scatter(y=df["relevancy"],        name="Relevancy (L3)",      line=dict(color="#f59e0b", width=2)))
fig.add_trace(go.Scatter(y=df["completeness"],     name="Completeness (L3)",   line=dict(color="#3b82f6", width=2)))
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

        golden = golden_map.get(q)
        if golden:
            with st.expander("📌 Golden Reference Answer (Ground Truth)", expanded=False):
                st.success(golden)
        st.markdown("---")

        for i, row in q_df.iterrows():
            run_label = f"**Run {i+1}**"
            contradicts = bool(row.get("contradicts_golden", 0))
            hallucinated = row.get("factual_hallucinated", "[]")
            try:
                hallucinated = json.loads(hallucinated) if isinstance(hallucinated, str) else hallucinated
            except Exception:
                hallucinated = []

            r1, r2, r3, r4, r5, r6, r7, r8 = st.columns([1, 3, 1, 1, 1, 1, 1, 1])

            r1.markdown(run_label)
            answer_text = row.get("answer", "")[:180]
            if contradicts:
                r2.caption(f"⚠ {answer_text}")
            else:
                r2.caption(answer_text)
            r3.markdown(score_badge(row.get("factual_anchor_score")), help=f"Hallucinated: {hallucinated[:2]}")
            r4.markdown(score_badge(row.get("golden_rouge_l")),       help="ROUGE-L vs golden answer")
            r5.markdown(score_badge(row.get("faithfulness")),         help=row.get("faithfulness_reason", ""))
            r6.markdown(score_badge(row.get("relevancy")),            help=row.get("relevancy_reason", ""))
            r7.markdown(score_badge(row.get("completeness")),         help=row.get("completeness_reason", ""))

            overall = row.get("overall_score")
            color = score_color(overall)
            bg = "#22c55e" if color == "green" else "#f59e0b" if color == "orange" else "#ef4444"
            r8.markdown(
                f'<div style="background:{bg};color:white;padding:4px 8px;'
                f'border-radius:6px;text-align:center;font-weight:700;">'
                f'{"N/A" if overall is None else f"{overall:.2f}"}</div>',
                unsafe_allow_html=True,
            )

            if contradicts:
                st.error(f"Run {i+1} contradicts golden answer: {row.get('contradiction_detail', '')}")
            if hallucinated:
                st.warning(f"Run {i+1} hallucinated facts not in source: {hallucinated}")

        st.markdown(
            "<div style='font-size:12px;color:#64748b;margin-top:4px;'>"
            "Columns: Run | Answer | 🔴Factual(L1) | 🟣GoldenROUGE(L2) | "
            "🟢Faithfulness(L3) | 🟡Relevancy(L3) | 🔵Completeness(L3) | Overall"
            "</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ── radar chart ───────────────────────────────────────────────────────────────

st.markdown("### 🎯 Average Score Breakdown")

radar_r     = [avg_faith, avg_relev, avg_compl,
               avg_factual if avg_factual is not None else 0,
               avg_golden_r if avg_golden_r is not None else 0,
               avg_overall]
radar_theta = ["Faithfulness (L3)", "Relevancy (L3)", "Completeness (L3)",
               "Factual Anchors (L1)", "Golden ROUGE-L (L2)", "Overall"]

fig2 = go.Figure(go.Scatterpolar(
    r=radar_r,
    theta=radar_theta,
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
