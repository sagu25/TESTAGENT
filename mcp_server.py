"""
RAG Evaluator MCP Server

Exposes 5 tools that any MCP client (Claude Code, Blueverse, etc.) can call:

  configure_agent   → set Blueverse credentials
  start_testing     → begin firing questions every N seconds
  stop_testing      → stop the testing loop
  get_status        → current scores and run count
  get_latest_results→ detailed last-N evaluation results

Run this server:
  python mcp_server.py

Then register the URL in Blueverse Foundry as an MCP tool server.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import threading
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

from mcp.server.fastmcp import FastMCP
import storage
import blueverse_connector
from agents.test_agent    import get_or_generate_questions, get_prioritized_questions
from agents.evaluator_agent import run as run_evaluator

mcp = FastMCP(
    name        = "RAG Evaluator",
    description = "Automated testing and evaluation for RAG-based AI agents.",
)

# ── Internal state ────────────────────────────────────────────────────────────

_scheduler_thread: threading.Thread | None = None
_stop_event       = threading.Event()
_config           = {
    "interval":    30,
    "target":      "blueverse",
    "configured":  False,
    "started_at":  None,
    "runs_this_session": 0,
}


def _run_one_cycle():
    """Run one test + evaluate cycle against the configured agent."""
    try:
        storage.init_db()
        questions = get_or_generate_questions()
        if not questions:
            return

        selected = get_prioritized_questions(questions, n=1)
        if not selected:
            selected = [questions[0]["question"] if isinstance(questions[0], dict)
                        else questions[0]]

        question = selected[0]
        print(f"[MCP] Firing: {question[:70]}")

        result = blueverse_connector.query(question)
        if not result:
            storage.save_to_dlq(question, "No response from Blueverse")
            return

        storage.save_test_run(
            question          = question,
            answer            = result.get("answer", ""),
            retrieved_context = result.get("retrieved_context", []),
            sources           = result.get("sources", []),
        )

        run_evaluator()
        _config["runs_this_session"] += 1
        print(f"[MCP] Cycle complete. Session runs: {_config['runs_this_session']}")

    except Exception as e:
        print(f"[MCP] Cycle error: {e}")


def _scheduler_loop(interval: int, stop_event: threading.Event):
    """Background thread — fires one cycle every `interval` seconds."""
    print(f"[MCP] Scheduler started. Interval: {interval}s")
    _run_one_cycle()  # run immediately on start

    while not stop_event.wait(timeout=interval):
        if stop_event.is_set():
            break
        _run_one_cycle()

    print("[MCP] Scheduler stopped.")


# ── MCP Tools ─────────────────────────────────────────────────────────────────

@mcp.tool()
def configure_agent(
    token_url:      str,
    chat_url:       str,
    client_id:      str,
    client_secret:  str,
    description:    str = "Employee policy AI assistant",
    request_field:  str = "message",
    response_field: str = "response",
    verify_ssl:     bool = True,
) -> str:
    """
    Configure the Blueverse agent credentials before starting testing.

    Parameters:
      token_url:      OAuth2 token endpoint URL
      chat_url:       Blueverse chat/query endpoint URL
      client_id:      OAuth2 client ID
      client_secret:  OAuth2 client secret
      description:    What the agent covers (used to generate test questions)
      request_field:  JSON field name for the question (default: message)
      response_field: JSON field name for the answer (default: response)
      verify_ssl:     Whether to verify SSL certificates (default: true)
    """
    blueverse_connector.configure(
        token_url      = token_url,
        chat_url       = chat_url,
        client_id      = client_id,
        client_secret  = client_secret,
        verify_ssl     = verify_ssl,
        request_field  = request_field,
        response_field = response_field,
    )

    os.environ["BLUEVERSE_DESCRIPTION"] = description
    os.environ["RAG_APP_URL"]           = "blueverse"

    # Clear old questions so new ones are generated for this agent
    try:
        storage.init_db()
        conn = storage.get_conn()
        conn.execute("DELETE FROM generated_questions")
        conn.execute("DELETE FROM golden_answers")
        conn.commit()
        conn.close()
    except Exception:
        pass

    _config["configured"] = True

    online = blueverse_connector.is_online()
    status = "ONLINE - connection successful" if online else "WARNING - could not connect, check credentials"

    return (
        f"Blueverse agent configured.\n"
        f"Chat URL:    {chat_url}\n"
        f"Description: {description}\n"
        f"Connection:  {status}\n\n"
        f"Now call start_testing() to begin evaluation."
    )


@mcp.tool()
def start_testing(interval_seconds: int = 30) -> str:
    """
    Start automated testing of the configured Blueverse agent.

    Fires one question every `interval_seconds` seconds (default: 30).
    Questions are auto-generated from the agent description.
    Each answer is evaluated across 3 layers:
      Layer 1: Factual accuracy (pure code)
      Layer 2: Similarity to golden reference (ROUGE-L)
      Layer 3: AI judge (faithfulness, relevancy, completeness)

    Parameters:
      interval_seconds: How often to fire a question (default: 30)
    """
    global _scheduler_thread, _stop_event

    if not _config["configured"]:
        return (
            "Agent not configured. Call configure_agent() first with "
            "Blueverse credentials."
        )

    if _scheduler_thread and _scheduler_thread.is_alive():
        return (
            f"Testing already running. "
            f"Runs this session: {_config['runs_this_session']}. "
            f"Call stop_testing() first to restart."
        )

    storage.init_db()

    _stop_event               = threading.Event()
    _config["interval"]       = interval_seconds
    _config["started_at"]     = datetime.now(timezone.utc).isoformat()
    _config["runs_this_session"] = 0

    _scheduler_thread = threading.Thread(
        target = _scheduler_loop,
        args   = (interval_seconds, _stop_event),
        daemon = True,
        name   = "mcp-evaluator",
    )
    _scheduler_thread.start()

    return (
        f"Testing started.\n"
        f"Interval:    Every {interval_seconds} seconds\n"
        f"Target:      Blueverse agent at {blueverse_connector.CHAT_URL}\n"
        f"First cycle: Running now\n\n"
        f"Call get_status() to see live scores.\n"
        f"Call stop_testing() to stop."
    )


@mcp.tool()
def stop_testing() -> str:
    """Stop the automated testing loop."""
    global _scheduler_thread

    if not _scheduler_thread or not _scheduler_thread.is_alive():
        return "Testing is not currently running."

    _stop_event.set()
    _scheduler_thread.join(timeout=10)

    runs = _config["runs_this_session"]
    started = _config.get("started_at", "unknown")

    return (
        f"Testing stopped.\n"
        f"Session runs:    {runs}\n"
        f"Started at:      {started}\n\n"
        f"Call get_latest_results() to see full evaluation report."
    )


@mcp.tool()
def get_status() -> str:
    """
    Get the current testing status and overall evaluation scores.
    Returns: run count, overall score, consistency, flagged questions.
    """
    try:
        storage.init_db()
        conn = storage.get_conn()

        total_runs = conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
        avg_row    = conn.execute(
            "SELECT AVG(overall_score), AVG(faithfulness), AVG(factual_anchor_score) "
            "FROM evaluations WHERE overall_score IS NOT NULL"
        ).fetchone()
        flagged    = conn.execute(
            "SELECT COUNT(*) FROM consistency_reports WHERE flagged = 1"
        ).fetchone()[0]
        contradicts = conn.execute(
            "SELECT COUNT(*) FROM evaluations WHERE contradicts_golden = 1"
        ).fetchone()[0]
        conn.close()

        running = _scheduler_thread and _scheduler_thread.is_alive()
        avg_overall = round(avg_row[0], 3) if avg_row[0] else 0
        avg_faith   = round(avg_row[1], 3) if avg_row[1] else 0
        avg_factual = round(avg_row[2], 3) if avg_row[2] else 0

        health = "GOOD" if avg_overall >= 0.80 else "WARNING" if avg_overall >= 0.60 else "POOR"

        return (
            f"=== RAG Evaluator Status ===\n"
            f"Testing running:      {'YES' if running else 'NO'}\n"
            f"Interval:             {_config['interval']}s\n"
            f"Runs this session:    {_config['runs_this_session']}\n"
            f"Total runs in DB:     {total_runs}\n\n"
            f"=== Scores ===\n"
            f"Overall Score:        {avg_overall} ({health})\n"
            f"Faithfulness:         {avg_faith}\n"
            f"Factual Anchors:      {avg_factual}\n"
            f"Flagged Inconsistent: {flagged}\n"
            f"Contradicts Golden:   {contradicts}\n\n"
            f"Score Guide: 0.80+ GOOD | 0.60-0.79 WARNING | below 0.60 POOR"
        )
    except Exception as e:
        return f"Error fetching status: {e}"


@mcp.tool()
def get_latest_results(last_n: int = 5) -> str:
    """
    Get detailed results of the last N evaluation runs.

    Parameters:
      last_n: Number of recent runs to show (default: 5)
    """
    try:
        storage.init_db()
        conn = storage.get_conn()
        rows = conn.execute(
            "SELECT e.question, e.overall_score, e.faithfulness, "
            "       e.factual_anchor_score, e.golden_rouge_l, "
            "       e.contradicts_golden, e.contradiction_detail, "
            "       tr.answer, e.timestamp "
            "FROM evaluations e "
            "JOIN test_runs tr ON e.run_id = tr.id "
            "ORDER BY e.timestamp DESC LIMIT ?",
            (last_n,)
        ).fetchall()
        conn.close()

        if not rows:
            return "No evaluation results yet. Start a testing session first."

        lines = [f"=== Last {last_n} Evaluation Results ===\n"]
        for i, row in enumerate(rows, 1):
            overall  = round(row[1], 2) if row[1] else "N/A"
            faith    = round(row[2], 2) if row[2] else "N/A"
            factual  = round(row[3], 2) if row[3] else "N/A"
            golden_r = round(row[4], 2) if row[4] else "N/A"
            contradicts = "YES" if row[5] else "NO"
            health   = "GOOD" if isinstance(overall, float) and overall >= 0.80 else \
                       "WARN" if isinstance(overall, float) and overall >= 0.60 else "POOR"

            lines.append(
                f"[{i}] {health} | Overall: {overall}\n"
                f"    Q: {row[0][:80]}\n"
                f"    A: {(row[7] or '')[:100]}\n"
                f"    Factual: {factual} | GoldenROUGE: {golden_r} | "
                f"Faithfulness: {faith} | Contradicts: {contradicts}\n"
                + (f"    Contradiction: {row[6]}\n" if row[5] and row[6] else "")
            )

        return "\n".join(lines)

    except Exception as e:
        return f"Error fetching results: {e}"


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("MCP_SERVER_PORT", "8502"))
    print(f"Starting RAG Evaluator MCP Server on port {port}...")
    print(f"Register this URL in Blueverse: http://localhost:{port}/sse")
    mcp.run(transport="sse")
