"""
LLM Response Cache — SQLite backed.

Key: SHA256(question + answer + golden_answer + eval_version)
Avoids re-calling the LLM for the same input across repeated evaluations.
Cache is invalidated when the golden answer changes (new key = new hash).
"""
import hashlib
import json
import sqlite3
import os
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "eval_results.db")


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_cache_table():
    conn = _conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_cache (
            cache_key  TEXT NOT NULL,
            metric     TEXT NOT NULL,
            result     TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (cache_key, metric)
        )
    """)
    conn.commit()
    conn.close()


def make_key(question: str, answer: str, golden: str, eval_version: str) -> str:
    content = f"{question}||{answer}||{golden}||{eval_version}"
    return hashlib.sha256(content.encode()).hexdigest()


def get(cache_key: str, metric: str) -> dict | None:
    try:
        conn = _conn()
        row = conn.execute(
            "SELECT result FROM llm_cache WHERE cache_key = ? AND metric = ?",
            (cache_key, metric),
        ).fetchone()
        conn.close()
        if row:
            return json.loads(row["result"])
    except Exception:
        pass
    return None


def set(cache_key: str, metric: str, result: dict):
    try:
        conn = _conn()
        conn.execute(
            """INSERT OR REPLACE INTO llm_cache (cache_key, metric, result, created_at)
               VALUES (?, ?, ?, ?)""",
            (cache_key, metric, json.dumps(result),
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def cache_stats() -> dict:
    try:
        conn = _conn()
        total = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
        by_metric = conn.execute(
            "SELECT metric, COUNT(*) as cnt FROM llm_cache GROUP BY metric"
        ).fetchall()
        conn.close()
        return {"total": total, "by_metric": {r["metric"]: r["cnt"] for r in by_metric}}
    except Exception:
        return {"total": 0, "by_metric": {}}
