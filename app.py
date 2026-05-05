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
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DB_PATH  = os.path.join(os.path.dirname(__file__), "eval_results.db")
RAG_URL  = os.getenv("RAG_APP_URL", "http://localhost:8000")

# ── Auto-testing global state (persists across Streamlit reruns) ──────────────
_auto_stop_event  : threading.Event | None = None
_auto_thread      : threading.Thread | None = None
_auto_run_count   = 0
_auto_started_at  : str | None = None


def _auto_loop(stop_event: threading.Event, interval: int):
    global _auto_run_count
    while not stop_event.is_set():
        try:
            from dotenv import load_dotenv as _ld
            _ld()
            import storage as _s
            _s.init_db()
            from agents.test_agent import run as _run_test
            from agents.evaluator_agent import run as _run_eval
            _run_test()
            _run_eval()
            _auto_run_count += 1
        except Exception as e:
            print(f"[AutoTest] Error: {e}")
        stop_event.wait(timeout=interval)


def is_auto_running() -> bool:
    return _auto_thread is not None and _auto_thread.is_alive()

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


def is_blueverse_mode() -> bool:
    return os.getenv("RAG_APP_URL", "http://localhost:8000").lower() == "blueverse"


def rag_online() -> bool:
    if is_blueverse_mode():
        try:
            import blueverse_connector
            return blueverse_connector.is_online()
        except Exception:
            return False
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
    if is_blueverse_mode():
        if online:
            st.success("Blueverse: Connected")
        else:
            st.error("Blueverse: Offline — check credentials + VPN")
    else:
        if online:
            st.success("RAG App: Online")
        else:
            st.error("RAG App: Offline — run `python start_rag_app.py`")

    st.divider()

    page = st.radio(
        "Navigate",
        ["📄 Documents", "💬 Chat", "🧪 Start Testing", "📊 Dashboard",
         "👁 Human Review", "⚔ Adversarial Questions", "📈 Regression",
         "📖 About Metrics"],
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
        if is_blueverse_mode():
            st.error("Blueverse is not reachable. Check credentials and VPN connection.")
        else:
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
        if is_blueverse_mode():
            st.error("Blueverse is not reachable. Check credentials and VPN connection.")
        else:
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

    # ── Auto Testing Section ──────────────────────────────────────────────────
    st.markdown("### Auto Testing")

    auto_col1, auto_col2, auto_col3 = st.columns([2, 1, 2])

    with auto_col1:
        auto_interval = st.slider(
            "Interval (seconds)", min_value=15, max_value=300,
            value=30, step=5,
            help="How often to fire a question automatically"
        )

    with auto_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if is_auto_running():
            st.success(f"Running — {_auto_run_count} runs")
        else:
            st.info("Stopped")

    with auto_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        start_auto = a1.button(
            "▶ Start Auto", use_container_width=True, type="primary",
            disabled=is_auto_running(),
            help=f"Fire 1 question every {auto_interval} seconds automatically"
        )
        stop_auto = a2.button(
            "⏹ Stop Auto", use_container_width=True,
            disabled=not is_auto_running(),
            help="Stop automatic testing"
        )

    if start_auto and not is_auto_running():
        _auto_run_count  = 0
        _auto_started_at = datetime.utcnow().strftime("%H:%M:%S")
        _auto_stop_event = threading.Event()
        _auto_thread     = threading.Thread(
            target=_auto_loop,
            args=(_auto_stop_event, auto_interval),
            daemon=True, name="streamlit-auto-test"
        )
        _auto_thread.start()
        st.success(f"Auto testing started — firing every {auto_interval}s. Dashboard updates automatically.")
        st.rerun()

    if stop_auto and is_auto_running():
        _auto_stop_event.set()
        st.warning(f"Auto testing stopped after {_auto_run_count} runs.")
        st.rerun()

    if is_auto_running():
        st.caption(
            f"Auto testing active since {_auto_started_at} | "
            f"Interval: {auto_interval}s | "
            f"Runs: {_auto_run_count} | "
            f"Dashboard refreshes every 30s automatically"
        )
        # Auto-refresh the page so scores update
        st.markdown('<meta http-equiv="refresh" content="30">', unsafe_allow_html=True)

    st.divider()

    # ── Manual Run Controls ───────────────────────────────────────────────────
    st.markdown("### Manual Testing")

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
    st.markdown("#### Layer 1 & 2 — Grounded Metrics (Zero LLM, Pure Math)")
    st.caption("These scores use no AI — they are calculated purely from code and mathematics. They cannot hallucinate.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Factual Anchors (L1)",
              f"{avg_factual:.2f}" if avg_factual is not None else "N/A",
              help="LAYER 1 — Pure Code. Extracts all numbers, percentages and dollar amounts from the source document and checks if the app's answer contains the same facts. Score 1.0 = every fact in the answer exists in the source. Score 0.0 = answer contains facts not in the document (hallucination). This is OUR ORIGINAL metric — not available in standard tools.")
    m2.metric("Golden ROUGE-L (L2)",
              f"{avg_golden:.2f}"  if avg_golden  is not None else "N/A",
              help="LAYER 2 — Pure Math (LCS formula). A verified reference answer is generated from the FULL document. ROUGE-L measures text overlap between the app's answer and this reference using Longest Common Subsequence. Score 1.0 = answer is very close to reference. Score 0.0 = answer shares almost nothing with reference. ROUGE-L is an INDUSTRY STANDARD metric used since 2004.")
    m3.metric("Contradicts Golden",
              contra_cnt,
              help="Number of runs where the app's answer directly contradicts the verified reference answer. Even if the answer sounds confident, if it contradicts the ground truth it is flagged here. Ideal = 0.")
    m4.metric("Inconsistent Questions",
              flagged_cnt,
              help="Number of questions where the app gave meaningfully different answers across multiple runs. Consistency score below 0.75 triggers this flag. Ideal = 0. High number means the app is unreliable.")

    st.markdown("#### Layer 3 — LLM Judge Metrics (AI-based, Grounded by Reference)")
    st.caption("These scores use an AI judge that is given the verified reference answer as a reference point — making it much harder for the judge itself to hallucinate.")
    m5, m6, m7, m8, m9 = st.columns(5)
    m5.metric("Overall Score",
              f"{avg_overall:.2f}",
              help="Final combined score using the formula: 25% Factual Anchors + 25% Golden ROUGE-L + 25% Faithfulness + 15% Relevancy + 10% Completeness. A hard cap of 0.45 is applied if Factual Anchors score below 0.30 — ensuring wrong facts always result in a poor overall score regardless of other metrics. Range: 0.0 to 1.0.")
    m6.metric("Faithfulness",
              f"{avg_faith:.2f}",
              help="INDUSTRY STANDARD metric (from RAGAS framework, 2023). Measures whether every claim in the app's answer is supported by the retrieved context AND consistent with the golden reference. Score 1.0 = fully faithful. Score 0.0 = answer contradicts the source. Our version is stronger than standard RAGAS because the judge also has the golden answer as a reference.")
    m7.metric("Relevancy",
              f"{avg_relev:.2f}",
              help="INDUSTRY STANDARD metric (from RAGAS framework, 2023). Measures whether the answer actually addresses the question asked. Score 1.0 = answer directly answers the question. Score 0.0 = answer is completely off-topic. An answer can be factually correct but irrelevant if it answers a different question.")
    m8.metric("Completeness",
              f"{avg_compl:.2f}",
              help="Measures whether the answer covers ALL important information from the golden reference answer. Score 1.0 = nothing important is missing. Score 0.0 = major information missing. Our version compares against the golden answer (full document), making it stricter than standard tools that only compare against retrieved chunks.")
    m9.metric("Total Runs",
              total_runs,
              help="Total number of question-answer pairs collected and evaluated so far. Each 'Run 1 Cycle' adds 3 runs. More runs = better consistency data and more reliable overall scores.")

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
            cons_map[row["question"]] = dict(row)

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


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — ABOUT METRICS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📖 About Metrics":
    st.markdown("## 📖 About the Evaluation Metrics")
    st.caption("A complete reference guide to every metric used in this system — what it measures, whether it is industry standard, the formula, and how to interpret the score.")
    st.divider()

    st.info("**How to read this page:** Each metric shows its origin (Standard / Our Design), the formula used, a real example, and what score range means good or bad performance.")

    # ── Overview ──────────────────────────────────────────────────────────────
    st.markdown("### System Overview — 3 Evaluation Layers")
    st.markdown("""
| Layer | Name | Method | AI Used? | Standard? |
|---|---|---|---|---|
| Layer 0 | Context Precision | LLM Judge | Yes | Industry Standard (RAGAS) |
| Layer 0 | Context Recall | LLM Judge | Yes | Industry Standard (RAGAS) |
| Layer 1 | Factual Anchor Score | Pure Code | **No** | **Our Original Design** |
| Layer 2 | Golden ROUGE-L | Math (LCS formula) | **No** | Standard formula, our application |
| Layer 3 | Faithfulness | LLM Judge + Golden | Yes | Industry Standard (RAGAS) |
| Layer 3 | Relevancy | LLM Judge + Golden | Yes | Industry Standard (RAGAS) |
| Layer 3 | Completeness | LLM Judge + Golden | Yes | Concept standard, our implementation |
| Extra | Consistency Score | LLM Pairwise + Math | Yes | **Our Original Design** |
| Final | Overall Score | Weighted Formula | No | **Our Original Design** |
""")

    st.divider()

    # ── Layer 0 ───────────────────────────────────────────────────────────────
    st.markdown("### Layer 0 — Retrieval Quality Metrics")
    st.caption("These metrics evaluate the RETRIEVAL step — before the LLM even generates an answer. They tell you whether the right document chunks were fetched.")

    with st.expander("📌 Context Precision — Industry Standard (RAGAS)", expanded=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**What it measures**")
            st.write("Of all the document chunks retrieved by the RAG system, how many are actually relevant to the question being asked?")
            st.markdown("**Origin**")
            st.write("Industry standard metric introduced by the RAGAS framework (2023). Used by RAGAS, DeepEval, LangSmith.")
            st.markdown("**Formula**")
            st.code("Context Precision = Relevant Chunks / Total Retrieved Chunks", language="text")
        with col2:
            st.markdown("**Example**")
            st.code("""Question: "How many vacation days?"

Retrieved:
  Chunk 1: "15 days annual leave"  ← RELEVANT
  Chunk 2: "leave accrues monthly" ← RELEVANT
  Chunk 3: "remote work policy"    ← NOT RELEVANT
  Chunk 4: "health insurance"      ← NOT RELEVANT

Precision = 2/4 = 0.50""", language="text")
            st.markdown("**Score Guide**")
            st.markdown("🟢 **0.80+** → Most retrieved chunks are relevant\n🟡 **0.50–0.79** → Mixed relevance\n🔴 **Below 0.50** → Retrieval is fetching wrong content")

    with st.expander("📌 Context Recall — Industry Standard (RAGAS)", expanded=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**What it measures**")
            st.write("Does the retrieved content contain ALL the information needed to fully and correctly answer the question? If key information is missing from the retrieved chunks, the LLM cannot give a complete answer even if it tries.")
            st.markdown("**Origin**")
            st.write("Industry standard metric from RAGAS framework (2023).")
            st.markdown("**Formula**")
            st.code("Context Recall = Info covered in retrieval / Total info needed to answer", language="text")
        with col2:
            st.markdown("**Example**")
            st.code("""Question: "What is the leave carry-forward policy?"

Golden answer needs:
  Fact 1: "up to 5 days carry-forward"
  Fact 2: "unused days beyond 5 lapse"
  Fact 3: "must be used by March 31"

Retrieved chunks contain: Fact 1 only

Recall = 1/3 = 0.33 ← critical info missing""", language="text")
            st.markdown("**Score Guide**")
            st.markdown("🟢 **0.80+** → Retrieved content covers all needed info\n🟡 **0.50–0.79** → Partial coverage\n🔴 **Below 0.50** → Retrieval is missing important chunks")

    st.divider()

    # ── Layer 1 ───────────────────────────────────────────────────────────────
    st.markdown("### Layer 1 — Factual Anchor Score")
    st.markdown("**Our Original Design — Not Available in Standard Tools**")

    with st.expander("🔴 Factual Anchor Score — Our Original Design", expanded=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**What it measures**")
            st.write("Checks whether the specific facts in the app's answer — numbers, percentages, dollar amounts, time periods — actually exist in the source document. This uses zero AI and zero LLM. It is pure Python regex code.")
            st.markdown("**Why we built this**")
            st.write("LLM judges can hallucinate. If an app says '99 days leave' and a standard LLM judge evaluates it, the judge might score it 0.80 if the sentence sounds natural. Our Layer 1 says '99 is not in the source document' — mathematically, with 100% reliability.")
            st.markdown("**Formula**")
            st.code("""Extract from SOURCE:  numbers, %, $, time facts
Extract from ANSWER:   numbers, %, $, time facts

Supported    = facts in answer that exist in source
Hallucinated = facts in answer NOT in source

Score = Supported / (Supported + Hallucinated)""", language="text")
        with col2:
            st.markdown("**Example — Wrong Answer Caught**")
            st.code("""Source: "15 days paid annual leave"
Answer: "99 days paid vacation"

Source facts:  ["15", "15 days"]
Answer facts:  ["99", "99 days"]

Supported:    0  (99 not in source)
Hallucinated: 1  (99 is fabricated)

Score = 0/1 = 0.00  CAUGHT IMMEDIATELY""", language="text")
            st.markdown("**Example — Correct Answer**")
            st.code("""Source: "80% of the health insurance premium"
Answer: "Company covers 80% of premium"

Supported:    1  (80% found in source)
Hallucinated: 0

Score = 1/1 = 1.00  CORRECT""", language="text")
            st.markdown("**Score Guide**")
            st.markdown("🟢 **1.00** → Every fact in answer exists in source\n🟡 **0.50–0.99** → Some facts unverified\n🔴 **Below 0.30** → Answer contains fabricated facts → Overall score hard-capped at 0.45")

    st.divider()

    # ── Layer 2 ───────────────────────────────────────────────────────────────
    st.markdown("### Layer 2 — Golden ROUGE-L Score")
    st.markdown("**Standard Formula (ROUGE since 2004), Our Application Against Golden Answer**")

    with st.expander("📐 Golden ROUGE-L — Standard Formula, Our Application", expanded=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**What it measures**")
            st.write("How similar is the app's answer to a verified reference answer generated from the FULL document? The Golden Answer is created by giving the LLM the complete policy document (not just retrieved chunks) — making it more complete and accurate than what the RAG app produces.")
            st.markdown("**What is ROUGE-L?**")
            st.write("ROUGE = Recall-Oriented Understudy for Gisting Evaluation. Introduced in 2004 by Chin-Yew Lin. Used by Google, Meta, and academic researchers worldwide for evaluating text generation quality.")
            st.markdown("**Formula**")
            st.code("""LCS = Longest Common Subsequence
       (longest sequence of words appearing in both texts
        in the same order, not necessarily consecutive)

Precision = LCS / length(app answer)
Recall    = LCS / length(golden answer)
ROUGE-L   = 2 × Precision × Recall / (Precision + Recall)""", language="text")
            st.markdown("**Why ROUGE-L not ROUGE-1 or ROUGE-2?**")
            st.code("""ROUGE-1: counts individual word overlaps
         → misses word order, too lenient
ROUGE-2: counts 2-word phrase overlaps
         → too strict, penalises paraphrasing
ROUGE-L: longest common subsequence
         → handles paraphrasing + word order
         → best balance for RAG evaluation""", language="text")
        with col2:
            st.markdown("**Example — Correct Answer (High Score)**")
            st.code("""Golden: "Full-time employees are entitled to
         15 days of paid annual leave per year"

Answer: "Employees receive 15 days of annual leave"

LCS = "employees ... 15 days ... annual leave"

Precision = 6/7  = 0.86
Recall    = 6/11 = 0.55
ROUGE-L   = 2×0.86×0.55 / (0.86+0.55) = 0.67""", language="text")
            st.markdown("**Example — Wrong Answer (Low Score)**")
            st.code("""Golden: "15 days of paid annual leave per year"

Answer: "Employees get 99 days vacation"

LCS = "employees" only (just 1 word)

ROUGE-L = very low (~0.08)
          Answer is far from the reference""", language="text")
            st.markdown("**Score Guide**")
            st.markdown("🟢 **0.70+** → Answer closely matches reference\n🟡 **0.40–0.69** → Partial match, possible gaps\n🔴 **Below 0.40** → Answer significantly different from reference")

    st.divider()

    # ── Layer 3 ───────────────────────────────────────────────────────────────
    st.markdown("### Layer 3 — LLM Judge Metrics")
    st.caption("These metrics use an AI model (Azure OpenAI / Groq) as a judge. Unlike standard tools, our judge is given the verified Golden Answer as a reference — making it significantly harder for the judge itself to hallucinate.")

    with st.expander("🤖 Faithfulness — Industry Standard (RAGAS), Enhanced by Us", expanded=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**What it measures**")
            st.write("Is every claim in the app's answer supported by the retrieved context AND consistent with the verified golden answer? Faithfulness catches hallucinations — when the app makes up facts not in the source.")
            st.markdown("**Origin**")
            st.write("Core metric from RAGAS framework (2023). Used by RAGAS, TruLens, DeepEval, LangSmith, Arize AI.")
            st.markdown("**Our Enhancement**")
            st.write("Standard RAGAS Faithfulness only uses retrieved context as reference. Our version also gives the judge the Golden Answer — so the judge has a verified ground truth to compare against, making it much harder to be fooled by confident-sounding wrong answers.")
        with col2:
            st.markdown("**Example**")
            st.code("""Context:  "Employees get 15 days leave"
Golden:   "15 days paid annual leave per year"
Answer:   "Employees get 30 days leave"

Judge asks:
  "Does answer contradict golden?"  YES
  "Is 30 days in the context?"      NO

Faithfulness = 0.05  LOW — correctly caught""", language="text")
            st.markdown("**Score Guide**")
            st.markdown("🟢 **0.80+** → Answer is grounded and faithful\n🟡 **0.50–0.79** → Some claims unverified\n🔴 **Below 0.50** → Answer contains hallucinations or contradictions")

    with st.expander("🤖 Relevancy — Industry Standard (RAGAS), Enhanced by Us", expanded=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**What it measures**")
            st.write("Does the answer actually address the question that was asked? An answer can be factually correct but completely irrelevant if it answers a different question.")
            st.markdown("**Origin**")
            st.write("Core metric from RAGAS framework (2023). Also called Answer Relevancy in standard tools.")
        with col2:
            st.markdown("**Example**")
            st.code("""Question: "How many vacation days?"
Answer A: "15 days paid leave"
          → Relevant → Score: 1.00

Answer B: "The company has remote work policy"
          → Off-topic → Score: 0.05

Answer C: "Leave policies vary"
          → Vague, avoids question → Score: 0.40""", language="text")
            st.markdown("**Score Guide**")
            st.markdown("🟢 **0.80+** → Directly answers the question\n🟡 **0.50–0.79** → Partially relevant\n🔴 **Below 0.50** → Off-topic or evasive")

    with st.expander("🤖 Completeness — Concept Standard, Our Implementation", expanded=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**What it measures**")
            st.write("Does the answer cover ALL important information from the golden reference? This catches answers that are correct but incomplete — missing key rules, limits, or conditions.")
            st.markdown("**Our Difference from Standard**")
            st.write("Standard tools compare completeness against retrieved context chunks. We compare against the Golden Answer (full document) — which is a stricter and more meaningful benchmark.")
        with col2:
            st.markdown("**Example**")
            st.code("""Question: "What is the leave carry-forward policy?"

Golden: "Up to 5 days carry-forward.
         Beyond 5 days lapses.
         Must use by March 31."

Answer: "You can carry forward unused leave."

Completeness = 0.15  ← missed 5-day limit,
                         lapse rule, deadline""", language="text")
            st.markdown("**Score Guide**")
            st.markdown("🟢 **0.80+** → Covers everything in the reference\n🟡 **0.50–0.79** → Some details missing\n🔴 **Below 0.50** → Major information omitted")

    st.divider()

    # ── Consistency ───────────────────────────────────────────────────────────
    st.markdown("### Consistency Score — Our Original Design")

    with st.expander("🔄 Consistency Score — Our Original Design", expanded=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**What it measures**")
            st.write("When the same question is asked multiple times across different test cycles, does the app give the same answer? Inconsistency means the app is unreliable — users asking the same question may get different (possibly contradictory) answers.")
            st.markdown("**Why this matters**")
            st.write("Most RAG evaluation tools evaluate one run at a time. They cannot tell you if the app is consistent over time. Our system stores every answer forever and compares them — catching drift, contradictions, and instability.")
            st.markdown("**Formula**")
            st.code("""For N answers to same question:

1. Pairwise Semantic Similarity
   sim(answer_i, answer_j) for all pairs
   avg_similarity = mean of all pairs

2. Contradiction Rate
   contradiction_rate = contradicting_pairs / total_pairs

3. Drift Score
   drift = 1 - sim(first_answer, latest_answer)

4. Final Score
   Consistency = avg_similarity
               × (1 - contradiction_rate)
               × (1 - 0.5 × drift)

Flagged as INCONSISTENT if score < 0.75""", language="text")
        with col2:
            st.markdown("**Example**")
            st.code("""Question: "How many vacation days?"

Run 1: "15 days paid leave"
Run 2: "15 annual leave days"
Run 3: "30 days vacation"     ← PROBLEM
Run 4: "15 days paid annually"

avg_similarity     = 0.51
contradiction_rate = 3/6 = 0.50
drift (Run1→Run4)  = 0.09

Consistency = 0.51 × 0.50 × 0.955
            = 0.24  FLAGGED""", language="text")
            st.markdown("**Score Guide**")
            st.markdown("🟢 **0.90+** → Same answer every time — very stable\n🟡 **0.75–0.89** → Minor phrasing variation — acceptable\n🔴 **Below 0.75** → Flagged — app giving different answers")

    st.divider()

    # ── Overall Formula ───────────────────────────────────────────────────────
    st.markdown("### Overall Score Formula — Our Original Design")

    with st.expander("⚖ Overall Score — Weighted Formula with Hard Cap", expanded=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Formula**")
            st.code("""Overall = 0.25 × Factual Anchor Score  (L1)
        + 0.25 × Golden ROUGE-L          (L2)
        + 0.25 × Faithfulness            (L3)
        + 0.15 × Relevancy               (L3)
        + 0.10 × Completeness            (L3)

HARD CAP RULE:
If Factual Anchor Score < 0.30
→ Overall is capped at maximum 0.45
  regardless of other scores""", language="text")
            st.markdown("**Why these weights?**")
            st.code("""Factual (0.25):    Most reliable — zero AI, deterministic
Golden ROUGE (0.25): Mathematical ground truth
Faithfulness (0.25): Core RAG health metric
Relevancy (0.15):    Important but secondary
Completeness (0.10): Partial answer > no answer""", language="text")
        with col2:
            st.markdown("**Why the Hard Cap?**")
            st.code("""Without hard cap:
  Factual: 0.10  (app said 99 days — WRONG)
  ROUGE-L: 0.15
  Faith:   0.90  (LLM judge was fooled)
  Relev:   0.95
  Compl:   0.80
  Overall: 0.51  ← YELLOW — misleadingly ok

With hard cap:
  Factual < 0.30 → cap at 0.45
  Overall: 0.45  ← RED — correctly flagged

Wrong facts must always produce a poor score.""", language="text")
            st.markdown("**Score Guide**")
            st.markdown("🟢 **0.80–1.00** → GOOD — App performing well\n🟡 **0.60–0.79** → WARNING — Some issues detected\n🔴 **0.00–0.59** → POOR — App has serious problems")

    st.divider()

    # ── Summary ───────────────────────────────────────────────────────────────
    st.markdown("### Standard vs Our Design — Quick Reference")
    st.markdown("""
| Metric | Standard or Ours | Based On |
|---|---|---|
| Context Precision | **Industry Standard** | RAGAS 2023 |
| Context Recall | **Industry Standard** | RAGAS 2023 |
| ROUGE-L formula | **Industry Standard** | Lin 2004, used by Google/Meta |
| Faithfulness | **Standard concept, enhanced** | RAGAS 2023 + our golden reference |
| Relevancy | **Standard concept, enhanced** | RAGAS 2023 + our golden reference |
| Completeness | **Standard concept, our impl.** | Inspired by DeepEval |
| Factual Anchor Score | **Our Original Design** | Built from scratch |
| Consistency Score | **Our Original Design** | Built from scratch |
| Evaluation Versioning | **Our Original Design** | Built from scratch |
| Overall Score Formula | **Our Original Design** | Built from scratch |
""")

    st.success("**The standard metrics (ROUGE-L, Faithfulness, Relevancy, Context Precision/Recall) validate our system against industry benchmarks. The original metrics (Factual Anchors, Consistency, Eval Versioning) address gaps that existing tools do not cover.**")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — HUMAN REVIEW
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "👁 Human Review":
    st.markdown("## 👁 Human Review")
    st.caption("Low-scoring answers (below 0.60) flagged here for manual review. "
               "Human verdicts override automated scores and improve golden answers.")
    st.divider()

    import storage as _st
    _st.init_db()

    threshold = st.slider("Flag answers below this score", 0.30, 0.80, 0.60, 0.05)
    pending   = _st.get_runs_pending_review(threshold)

    if not pending:
        st.success(f"No answers below {threshold:.2f} awaiting review.")
    else:
        st.warning(f"{len(pending)} answer(s) need human review.")

    for run in pending:
        run_id   = run["id"]
        question = run["question"]
        answer   = run["answer"]
        score    = run.get("overall_score", 0)

        with st.expander(
            f"[Score: {score:.2f}] {question[:70]}...",
            expanded=True,
        ):
            st.markdown(f"**Question:** {question}")
            st.markdown(f"**App Answer:** {answer}")

            gold_map2 = load_golden_answers()
            golden = gold_map2.get(question, "")
            if golden:
                st.info(f"**Golden Answer:** {golden}")

            c1, c2 = st.columns(2)
            with c1:
                verdict = st.selectbox(
                    "Verdict",
                    ["Correct", "Partially Correct", "Incorrect"],
                    key=f"v_{run_id}",
                )
                human_score = {"Correct": 1.0, "Partially Correct": 0.5, "Incorrect": 0.0}[verdict]
                reviewer = st.text_input("Your name", value="Reviewer", key=f"rev_{run_id}")

            with c2:
                notes = st.text_area("Notes / Reason", key=f"n_{run_id}",
                                     placeholder="Why is this correct/incorrect?")
                new_golden = st.text_area(
                    "Edit Golden Answer (optional — updates ground truth)",
                    value=golden, key=f"g_{run_id}", height=100,
                )

            if st.button(f"Submit Review for Run {run_id}", key=f"sub_{run_id}",
                         type="primary"):
                _st.save_human_review(run_id, question, answer,
                                      verdict, human_score, notes, reviewer)
                if new_golden and new_golden != golden:
                    _st.update_golden_answer(question, new_golden)
                    st.success("Review saved + Golden answer updated.")
                else:
                    st.success("Review saved.")
                st.rerun()

    st.divider()
    st.markdown("### Completed Reviews")
    reviews = _st.get_all_human_reviews()
    if reviews:
        rev_df = pd.DataFrame(reviews)[
            ["question", "verdict", "human_score", "notes", "reviewed_by", "reviewed_at"]
        ]
        rev_df["question"] = rev_df["question"].str[:60] + "..."
        st.dataframe(rev_df, use_container_width=True, hide_index=True)
    else:
        st.info("No completed reviews yet.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ADVERSARIAL QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "⚔ Adversarial Questions":
    st.markdown("## ⚔ Adversarial & Manual Questions")
    st.caption("Add your own test questions — edge cases, trick questions, out-of-scope queries. "
               "These are added to the test rotation alongside auto-generated questions.")
    st.divider()

    import storage as _st2
    _st2.init_db()

    st.markdown("### Add a Question")
    q_col1, q_col2 = st.columns([3, 1])
    with q_col1:
        new_q = st.text_area("Question", height=80,
                             placeholder="e.g. Can an employee take 200 days of leave?")
    with q_col2:
        q_type = st.selectbox("Type", [
            "adversarial",
            "edge case",
            "out of scope",
            "factual",
            "trick",
        ])
        q_cat = st.text_input("Category", value="manual")
        expected = st.text_input("Expected answer hint (optional)")

    if st.button("Add Question", type="primary", use_container_width=True):
        if new_q.strip():
            _st2.save_manual_question(new_q.strip(), q_cat, q_type, expected)
            st.success("Question added to test rotation.")
            st.rerun()
        else:
            st.warning("Please enter a question.")

    st.divider()
    st.markdown("### Question Bank")

    auto_qs   = _st2.get_generated_questions()
    manual_qs = _st2.get_manual_questions()

    tab1, tab2 = st.tabs([f"Auto-Generated ({len(auto_qs)})", f"Manual ({len(manual_qs)})"])

    with tab1:
        for q in auto_qs:
            cat = q.get("category", "general")
            st.markdown(f"- `[{cat}]` {q['question']}")

    with tab2:
        if not manual_qs:
            st.info("No manual questions added yet.")
        for q in manual_qs:
            c1, c2 = st.columns([5, 1])
            c1.markdown(
                f"- `[{q['question_type']}]` {q['question']}"
                + (f"\n  *Expected: {q['expected_answer']}*" if q.get("expected_answer") else "")
            )
            if c2.button("Remove", key=f"rm_{q['id']}"):
                _st2.delete_manual_question(q["id"])
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — REGRESSION TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Regression":
    st.markdown("## 📈 Regression Tracking")
    st.caption("Take named snapshots of current scores. Compare any two snapshots to detect regressions.")
    st.divider()

    import storage as _st3
    _st3.init_db()

    # Take snapshot
    st.markdown("### Take Snapshot")
    snap_name = st.text_input("Snapshot name",
                               placeholder="e.g. v1.0-baseline, after-doc-update, post-fix")
    if st.button("📸 Save Snapshot", type="primary"):
        if snap_name.strip():
            snap = _st3.take_snapshot(snap_name.strip())
            st.success(
                f"Snapshot '{snap_name}' saved — "
                f"Overall: {snap['avg_overall']:.2f} | "
                f"Faithfulness: {snap['avg_faithfulness']:.2f} | "
                f"Runs: {snap['total_runs']}"
            )
            st.rerun()
        else:
            st.warning("Enter a snapshot name.")

    st.divider()

    snapshots = _st3.get_snapshots()
    if not snapshots:
        st.info("No snapshots yet. Run some test cycles then save a snapshot.")
        st.stop()

    snap_df = pd.DataFrame(snapshots)

    # Comparison
    st.markdown("### Compare Two Snapshots")
    snap_names = [s["name"] for s in snapshots]

    if len(snapshots) >= 2:
        cmp1, cmp2 = st.columns(2)
        with cmp1:
            base_name = st.selectbox("Baseline", snap_names, index=len(snap_names)-2)
        with cmp2:
            curr_name = st.selectbox("Current",  snap_names, index=len(snap_names)-1)

        base = next(s for s in snapshots if s["name"] == base_name)
        curr = next(s for s in snapshots if s["name"] == curr_name)

        metrics_to_compare = [
            ("Overall Score",      "avg_overall"),
            ("Faithfulness",       "avg_faithfulness"),
            ("Relevancy",          "avg_relevancy"),
            ("Completeness",       "avg_completeness"),
            ("Factual Anchors",    "avg_factual"),
            ("Golden ROUGE-L",     "avg_golden_rouge"),
        ]

        st.markdown(f"#### {base_name}  →  {curr_name}")
        cols = st.columns(len(metrics_to_compare))
        for col, (label, key) in zip(cols, metrics_to_compare):
            b_val = base.get(key) or 0
            c_val = curr.get(key) or 0
            delta = round(c_val - b_val, 3)
            col.metric(label, f"{c_val:.2f}", delta=f"{delta:+.3f}",
                       delta_color="normal")

        st.markdown("---")
        flag_delta = (curr.get("flagged_questions", 0) or 0) - (base.get("flagged_questions", 0) or 0)
        cont_delta = (curr.get("contradiction_count", 0) or 0) - (base.get("contradiction_count", 0) or 0)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Runs (base)",    base.get("total_runs", 0))
        c2.metric("Total Runs (current)", curr.get("total_runs", 0))
        c3.metric("Inconsistent Qs", curr.get("flagged_questions", 0),
                  delta=f"{flag_delta:+d}", delta_color="inverse")
        c4.metric("Contradictions",  curr.get("contradiction_count", 0),
                  delta=f"{cont_delta:+d}", delta_color="inverse")
    else:
        st.info("Need at least 2 snapshots to compare. Run more cycles and save another snapshot.")

    st.divider()

    # Snapshot history chart
    st.markdown("### Score History Across Snapshots")
    if len(snap_df) > 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=snap_df["name"], y=snap_df["avg_overall"],
                                 name="Overall", line=dict(color="#6366f1", width=3), mode="lines+markers"))
        fig.add_trace(go.Scatter(x=snap_df["name"], y=snap_df["avg_faithfulness"],
                                 name="Faithfulness", line=dict(color="#22c55e", width=2), mode="lines+markers"))
        fig.add_trace(go.Scatter(x=snap_df["name"], y=snap_df["avg_factual"],
                                 name="Factual Anchors", line=dict(color="#dc2626", width=2), mode="lines+markers"))
        fig.add_hline(y=0.75, line_dash="dash", line_color="red", annotation_text="Threshold 0.75")
        fig.update_layout(
            height=320, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Snapshot", yaxis=dict(range=[0, 1.05]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # All snapshots table
    st.markdown("### All Snapshots")
    display_cols = ["name", "total_runs", "avg_overall", "avg_faithfulness",
                    "avg_factual", "flagged_questions", "created_at"]
    avail = [c for c in display_cols if c in snap_df.columns]
    st.dataframe(snap_df[avail].rename(columns={
        "name": "Snapshot", "total_runs": "Runs",
        "avg_overall": "Overall", "avg_faithfulness": "Faithfulness",
        "avg_factual": "Factual", "flagged_questions": "Flagged",
        "created_at": "Saved At",
    }), use_container_width=True, hide_index=True)
