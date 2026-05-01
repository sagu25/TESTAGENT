import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import sqlite3
import requests
import threading
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DB_PATH  = os.path.join(os.path.dirname(__file__), "eval_results.db")
RAG_URL  = os.getenv("RAG_APP_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RAG Evaluation System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── helpers ───────────────────────────────────────────────────────────────────

def get_conn():
    if not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def rag_online() -> bool:
    try:
        r = requests.get(f"{RAG_URL}/", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def score_color(score):
    if score is None: return "#94a3b8"
    if score >= 0.80: return "#22c55e"
    if score >= 0.60: return "#f59e0b"
    return "#ef4444"


def score_badge(score):
    if score is None: return "⚪ N/A"
    if score >= 0.80: return f"🟢 {score:.2f}"
    if score >= 0.60: return f"🟡 {score:.2f}"
    return f"🔴 {score:.2f}"


def extract_pdf_text(file_bytes: bytes) -> str:
    try:
        import pypdf
        import io
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        return f"Error reading PDF: {e}"


def load_evaluations() -> pd.DataFrame:
    conn = get_conn()
    if not conn:
        return pd.DataFrame()
    rows = conn.execute("""
        SELECT e.id, e.run_id, e.question, e.faithfulness, e.relevancy,
               e.completeness, e.rouge_l, e.overall_score,
               e.faithfulness_reason, e.relevancy_reason, e.completeness_reason,
               e.factual_anchor_score, e.factual_hallucinated,
               e.golden_rouge_l, e.contradicts_golden, e.contradiction_detail,
               tr.answer, e.timestamp
        FROM evaluations e
        JOIN test_runs tr ON e.run_id = tr.id
        ORDER BY e.timestamp ASC
    """).fetchall()
    conn.close()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def load_consistency() -> pd.DataFrame:
    conn = get_conn()
    if not conn:
        return pd.DataFrame()
    rows = conn.execute(
        "SELECT * FROM consistency_reports ORDER BY consistency_score ASC"
    ).fetchall()
    conn.close()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def load_golden_answers() -> dict:
    conn = get_conn()
    if not conn:
        return {}
    rows = conn.execute("SELECT question, golden_answer FROM golden_answers").fetchall()
    conn.close()
    return {r["question"]: r["golden_answer"] for r in rows}


def run_pipeline_once():
    """Run one test + evaluate cycle (called from button)."""
    import storage
    storage.init_db()
    from agents.test_agent import run as run_test
    from agents.evaluator_agent import run as run_eval
    run_test()
    run_eval()


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 RAG Eval System")
    online = rag_online()
    if online:
        st.success("RAG App: Online")
    else:
        st.error("RAG App: Offline — run `python start_rag_app.py`")

    st.divider()

    page = st.radio(
        "Navigate",
        ["📄 Documents", "💬 Chat", "🧪 Start Testing", "📊 Dashboard"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(f"Provider: **{os.getenv('LLM_PROVIDER','azure').upper()}**")
    st.caption(f"RAG: `{RAG_URL}`")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DOCUMENTS
# ═══════════════════════════════════════════════════════════════════════════════

if page == "📄 Documents":
    st.markdown("## 📄 Policy Documents")
    st.caption("Upload your own policy documents. The RAG app will use them immediately.")
    st.divider()

    # Upload section
    st.markdown("### Upload New Document")
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload a policy document",
            type=["pdf", "txt"],
            label_visibility="collapsed",
        )

    with col2:
        doc_title = st.text_input("Document Title", placeholder="e.g. Travel Policy")

    if uploaded_file and doc_title:
        if st.button("➕ Add Document", use_container_width=True, type="primary"):
            with st.spinner("Processing document..."):
                if uploaded_file.name.endswith(".pdf"):
                    content = extract_pdf_text(uploaded_file.read())
                else:
                    content = uploaded_file.read().decode("utf-8", errors="ignore")

                if content and not content.startswith("Error"):
                    from rag_app.document_store import add_document, clear_generated_questions
                    add_document(doc_title, content, uploaded_file.name)

                    try:
                        requests.post(f"{RAG_URL}/reload", timeout=5)
                    except Exception:
                        pass

                    clear_generated_questions()
                    st.success(f"'{doc_title}' added successfully. Questions will be regenerated on next test run.")
                    st.rerun()
                else:
                    st.error(f"Could not read document: {content}")
    elif uploaded_file and not doc_title:
        st.warning("Please enter a document title.")

    st.divider()

    # Current documents
    st.markdown("### Loaded Documents")
    try:
        resp = requests.get(f"{RAG_URL}/documents", timeout=5)
        if resp.status_code == 200:
            docs = resp.json()["documents"]
            for doc in docs:
                badge = "🟢 Default" if doc["type"] == "default" else "🔵 Uploaded"
                with st.expander(f"{badge} — {doc['title']}"):
                    st.caption(doc["preview"])
                    if doc["type"] == "uploaded":
                        if st.button(f"🗑 Remove '{doc['title']}'", key=f"del_{doc['title']}"):
                            from rag_app.document_store import remove_document, clear_generated_questions
                            remove_document(doc["title"])
                            try:
                                requests.post(f"{RAG_URL}/reload", timeout=5)
                            except Exception:
                                pass
                            clear_generated_questions()
                            st.rerun()
        else:
            st.info("Start the RAG app to view loaded documents.")
    except Exception:
        st.info("Start the RAG app to view loaded documents (`python start_rag_app.py`)")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CHAT
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "💬 Chat":
    st.markdown("## 💬 Chat with RAG App")
    st.caption("Ask any question. The RAG app retrieves relevant policy sections and answers.")
    st.divider()

    if not online:
        st.error("RAG App is offline. Please run `python start_rag_app.py` first.")
        st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.chat_input("Ask a policy question...")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                st.caption(f"Sources: {', '.join(msg['sources'])}")
                with st.expander("View retrieved context"):
                    for chunk in msg.get("context", []):
                        st.markdown(f"**[{chunk['source']}]** — score: {chunk['score']:.2f}")
                        st.caption(chunk["text"][:300])

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Searching policy documents..."):
                try:
                    resp = requests.post(
                        f"{RAG_URL}/query",
                        json={"question": question},
                        timeout=30,
                    )
                    data = resp.json()
                    answer  = data.get("answer", "No answer returned.")
                    sources = data.get("sources", [])
                    context = data.get("retrieved_context", [])
                    st.markdown(answer)
                    st.caption(f"Sources: {', '.join(sources) if sources else 'None found'}")
                    with st.expander("View retrieved context"):
                        for chunk in context:
                            st.markdown(f"**[{chunk['source']}]** — score: {chunk['score']:.2f}")
                            st.caption(chunk["text"][:300])
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": answer,
                        "sources": sources, "context": context,
                    })
                except Exception as e:
                    st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — START TESTING
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🧪 Start Testing":
    st.markdown("## 🧪 Start Testing")
    st.caption("Trigger a test + evaluation cycle manually. Each run fires one question and evaluates the answer.")
    st.divider()

    if not online:
        st.error("RAG App is offline. Please run `python start_rag_app.py` first.")
        st.stop()

    # Show generated questions
    conn = get_conn()
    questions = []
    if conn:
        rows = conn.execute("SELECT question, category FROM generated_questions").fetchall()
        questions = [dict(r) for r in rows]
        conn.close()

    col1, col2 = st.columns([3, 1])
    with col1:
        if questions:
            st.markdown(f"**{len(questions)} test questions ready** (auto-generated from your documents)")
            for i, q in enumerate(questions):
                cat = q.get("category", "general")
                st.markdown(f"`{i+1}.` [{cat}] {q['question']}")
        else:
            st.info("No questions generated yet. Questions will be auto-generated from your documents on first test run.")

    with col2:
        total_runs = 0
        if conn:
            try:
                conn2 = get_conn()
                total_runs = conn2.execute("SELECT COUNT(*) FROM test_runs").fetchone()[0]
                conn2.close()
            except Exception:
                pass
        st.metric("Total Runs So Far", total_runs)

    st.divider()

    # Run controls
    c1, c2, c3 = st.columns(3)

    with c1:
        run_once = st.button("▶ Run 1 Cycle", use_container_width=True, type="primary",
                             help="Fire 1 question, evaluate, update report")
    with c2:
        run_five = st.button("⏩ Run 5 Cycles", use_container_width=True,
                             help="Fire 5 questions in sequence")
    with c3:
        reset_btn = st.button("🔄 Reset Questions", use_container_width=True,
                              help="Delete generated questions — system will re-analyze app")

    if reset_btn:
        from rag_app.document_store import clear_generated_questions
        clear_generated_questions()
        st.success("Questions cleared. Next test run will regenerate them from your documents.")
        st.rerun()

    if run_once:
        log = st.empty()
        progress = st.progress(0)
        with st.spinner("Running test + evaluation..."):
            log.info("Step 1/3: Test Agent firing question...")
            progress.progress(20)
            try:
                run_pipeline_once()
                progress.progress(100)
                log.success("Cycle complete! Go to Dashboard to see results.")
                st.balloons()
            except Exception as e:
                st.error(f"Error during pipeline: {e}")

    if run_five:
        progress = st.progress(0)
        status   = st.empty()
        try:
            for i in range(5):
                status.info(f"Running cycle {i+1}/5...")
                run_pipeline_once()
                progress.progress((i + 1) * 20)
            status.success("5 cycles complete! Go to Dashboard to see results.")
            st.balloons()
        except Exception as e:
            st.error(f"Error at cycle: {e}")

    st.divider()

    # Recent test runs
    st.markdown("### Recent Test Runs")
    df = load_evaluations()
    if not df.empty:
        recent = df.tail(10)[["question", "overall_score", "faithfulness",
                               "factual_anchor_score", "timestamp"]].copy()
        recent.columns = ["Question", "Overall", "Faithfulness", "Factual Anchors", "Time"]
        recent["Question"] = recent["Question"].str[:70] + "..."
        recent["Time"] = recent["Time"].str[:19]
        st.dataframe(recent, use_container_width=True, hide_index=True)
    else:
        st.info("No runs yet. Click 'Run 1 Cycle' to start.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Dashboard":
    st.markdown("## 📊 Evaluation Dashboard")
    st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    col_r, col_dl = st.columns([5, 1])
    with col_r:
        if st.button("🔄 Refresh"):
            st.rerun()
    with col_dl:
        report_path = os.path.join(os.path.dirname(__file__), "reports", "report.html")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.download_button(
                "⬇ Download Report",
                data=html_content,
                file_name=f"rag_eval_report_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.html",
                mime="text/html",
                use_container_width=True,
            )

    st.divider()

    df       = load_evaluations()
    cons_df  = load_consistency()
    gold_map = load_golden_answers()

    if df.empty:
        st.info("No evaluation data yet. Go to **Start Testing** and run a cycle first.")
        st.stop()

    total_runs  = len(df)
    unique_qs   = df["question"].nunique()
    avg_overall = df["overall_score"].mean()
    avg_faith   = df["faithfulness"].mean()
    avg_relev   = df["relevancy"].mean()
    avg_compl   = df["completeness"].mean()
    avg_factual = df["factual_anchor_score"].mean() if "factual_anchor_score" in df.columns and df["factual_anchor_score"].notna().any() else None
    avg_golden  = df["golden_rouge_l"].mean() if "golden_rouge_l" in df.columns and df["golden_rouge_l"].notna().any() else None
    flagged_cnt = int(cons_df["flagged"].sum()) if not cons_df.empty else 0
    contra_cnt  = int(df["contradicts_golden"].sum()) if "contradicts_golden" in df.columns else 0

    # Metric cards
    st.markdown("#### Grounded Metrics (Zero LLM)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Factual Anchor Score", f"{avg_factual:.2f}" if avg_factual is not None else "N/A")
    m2.metric("Golden ROUGE-L",       f"{avg_golden:.2f}"  if avg_golden  is not None else "N/A")
    m3.metric("Contradicts Golden",   contra_cnt)
    m4.metric("Inconsistent Questions", flagged_cnt)

    st.markdown("#### LLM Judge Metrics (Grounded)")
    m5, m6, m7, m8, m9 = st.columns(5)
    m5.metric("Overall Score", f"{avg_overall:.2f}")
    m6.metric("Faithfulness",  f"{avg_faith:.2f}")
    m7.metric("Relevancy",     f"{avg_relev:.2f}")
    m8.metric("Completeness",  f"{avg_compl:.2f}")
    m9.metric("Total Runs",    total_runs)

    st.divider()

    # Score trend
    st.markdown("### Score Trend Across All Runs")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df["overall_score"], name="Overall",
                             line=dict(color="#6366f1", width=3)))
    if "factual_anchor_score" in df.columns:
        fig.add_trace(go.Scatter(y=df["factual_anchor_score"], name="Factual Anchors (L1)",
                                 line=dict(color="#dc2626", width=2)))
    if "golden_rouge_l" in df.columns:
        fig.add_trace(go.Scatter(y=df["golden_rouge_l"], name="Golden ROUGE-L (L2)",
                                 line=dict(color="#7c3aed", width=2)))
    fig.add_trace(go.Scatter(y=df["faithfulness"], name="Faithfulness",
                             line=dict(color="#22c55e", width=2)))
    fig.add_trace(go.Scatter(y=df["relevancy"],    name="Relevancy",
                             line=dict(color="#f59e0b", width=2)))
    fig.add_hline(y=0.75, line_dash="dash", line_color="red",
                  annotation_text="Threshold 0.75")
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      yaxis=dict(range=[0, 1.05]),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Consistency alerts
    if not cons_df.empty:
        flagged = cons_df[cons_df["flagged"] == 1]
        if not flagged.empty:
            st.markdown("### Consistency Alerts")
            for _, row in flagged.iterrows():
                with st.expander(
                    f"🔴 {row['question'][:80]}... — Consistency: {row['consistency_score']:.2f}",
                    expanded=False
                ):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Consistency", f"{row['consistency_score']:.2f}")
                    c2.metric("Contradiction Rate", f"{row['contradiction_rate']*100:.0f}%")
                    c3.metric("Drift", f"{row['drift_score']:.2f}")
                    details = row.get("contradiction_details", "[]")
                    if isinstance(details, str):
                        try:
                            details = json.loads(details)
                        except Exception:
                            details = []
                    for d in details:
                        st.error(f"Run {d.get('run_a')} vs Run {d.get('run_b')}: {d.get('detail', '')}")
            st.divider()

    # Per-question comparison
    st.markdown("### Per-Question Answer Comparison")
    questions_list = df["question"].unique()
    cons_map = {}
    if not cons_df.empty:
        for _, row in cons_df.iterrows():
            cons_map[row["question"]] = row

    for q in questions_list:
        q_df   = df[df["question"] == q].reset_index(drop=True)
        cons   = cons_map.get(q, {})
        cs     = cons.get("consistency_score") if cons else None
        flagged = bool(cons.get("flagged", False)) if cons else False

        label = f"{'🔴 ' if flagged else '✅ '}{q[:80]}..."
        cs_text = f" | Consistency: {cs:.2f}" if cs is not None else ""

        with st.expander(f"{label}{cs_text} ({len(q_df)} runs)", expanded=flagged):

            if cs is not None:
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.metric("Consistency",       f"{cs:.2f}")
                cc2.metric("Contradiction Rate", f"{cons.get('contradiction_rate',0)*100:.0f}%")
                cc3.metric("Drift",              f"{cons.get('drift_score',0):.2f}")
                cc4.metric("Runs",               len(q_df))

            golden = gold_map.get(q)
            if golden:
                with st.expander("📌 Golden Reference Answer", expanded=False):
                    st.success(golden)
            st.markdown("---")

            for i, row in q_df.iterrows():
                contradicts  = bool(row.get("contradicts_golden", 0))
                hallucinated = row.get("factual_hallucinated", "[]")
                try:
                    hallucinated = json.loads(hallucinated) if isinstance(hallucinated, str) else hallucinated
                except Exception:
                    hallucinated = []

                r1, r2, r3, r4, r5, r6, r7, r8 = st.columns([1, 3, 1, 1, 1, 1, 1, 1])
                r1.markdown(f"**Run {i+1}**")
                answer_text = (row.get("answer") or "")[:180]
                r2.caption(f"⚠ {answer_text}" if contradicts else answer_text)
                r3.markdown(score_badge(row.get("factual_anchor_score")),
                            help=f"Hallucinated: {hallucinated[:2]}")
                r4.markdown(score_badge(row.get("golden_rouge_l")),
                            help="ROUGE-L vs golden")
                r5.markdown(score_badge(row.get("faithfulness")),
                            help=row.get("faithfulness_reason", ""))
                r6.markdown(score_badge(row.get("relevancy")),
                            help=row.get("relevancy_reason", ""))
                r7.markdown(score_badge(row.get("completeness")),
                            help=row.get("completeness_reason", ""))
                o  = row.get("overall_score")
                bg = score_color(o)
                r8.markdown(
                    f'<div style="background:{bg};color:white;padding:4px 8px;'
                    f'border-radius:6px;text-align:center;font-weight:700;">'
                    f'{"N/A" if o is None else f"{o:.2f}"}</div>',
                    unsafe_allow_html=True,
                )

                if contradicts:
                    st.error(f"Run {i+1} contradicts golden: {row.get('contradiction_detail','')}")
                if hallucinated:
                    st.warning(f"Run {i+1} hallucinated facts: {hallucinated}")

            st.caption(
                "Columns: Run | Answer | Factual(L1) | GoldenROUGE(L2) | "
                "Faithfulness | Relevancy | Completeness | Overall"
            )

    st.divider()

    # Radar chart
    st.markdown("### Score Breakdown")
    radar_r = [avg_faith, avg_relev, avg_compl,
               avg_factual if avg_factual else 0,
               avg_golden if avg_golden else 0,
               avg_overall]
    radar_t = ["Faithfulness", "Relevancy", "Completeness",
               "Factual Anchors", "Golden ROUGE-L", "Overall"]
    fig2 = go.Figure(go.Scatterpolar(
        r=radar_r, theta=radar_t, fill="toself",
        line_color="#6366f1", fillcolor="rgba(99,102,241,0.2)",
    ))
    fig2.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=350, margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Raw table
    with st.expander("📋 Raw Data Table"):
        cols = ["run_id", "question", "factual_anchor_score", "golden_rouge_l",
                "faithfulness", "relevancy", "completeness", "overall_score", "timestamp"]
        available = [c for c in cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True, hide_index=True)

    st.caption("Built with Streamlit + Azure OpenAI | 3-Layer Grounded Evaluation")
