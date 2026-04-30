import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "eval_results.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS test_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            retrieved_context TEXT NOT NULL,
            sources TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            evaluated INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            faithfulness REAL,
            relevancy REAL,
            completeness REAL,
            rouge_l REAL,
            overall_score REAL,
            faithfulness_reason TEXT,
            relevancy_reason TEXT,
            completeness_reason TEXT,
            factual_anchor_score REAL,
            factual_supported TEXT,
            factual_hallucinated TEXT,
            golden_rouge_l REAL,
            contradicts_golden INTEGER DEFAULT 0,
            contradiction_detail TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES test_runs(id)
        );

        CREATE TABLE IF NOT EXISTS golden_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL UNIQUE,
            golden_answer TEXT NOT NULL,
            factual_anchors TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS consistency_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            consistency_score REAL,
            contradiction_rate REAL,
            drift_score REAL,
            total_runs INTEGER,
            flagged INTEGER DEFAULT 0,
            contradiction_details TEXT,
            timestamp TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS generated_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            category TEXT,
            created_at TEXT NOT NULL
        );
    """)
    conn.commit()

    # Migrate existing evaluations table — add new columns if missing
    new_cols = [
        ("factual_anchor_score",  "REAL"),
        ("factual_supported",     "TEXT"),
        ("factual_hallucinated",  "TEXT"),
        ("golden_rouge_l",        "REAL"),
        ("contradicts_golden",    "INTEGER DEFAULT 0"),
        ("contradiction_detail",  "TEXT"),
    ]
    for col, col_type in new_cols:
        try:
            conn.execute(f"ALTER TABLE evaluations ADD COLUMN {col} {col_type}")
            conn.commit()
        except Exception:
            pass  # column already exists

    conn.close()


def save_test_run(question: str, answer: str, retrieved_context: list, sources: list) -> int:
    conn = get_conn()
    cursor = conn.execute(
        """INSERT INTO test_runs (question, answer, retrieved_context, sources, timestamp)
           VALUES (?, ?, ?, ?, ?)""",
        (question, answer, json.dumps(retrieved_context), json.dumps(sources),
         datetime.utcnow().isoformat()),
    )
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return run_id


def get_unevaluated_runs() -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM test_runs WHERE evaluated = 0 ORDER BY timestamp ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_evaluation(run_id: int, question: str, scores: dict):
    conn = get_conn()
    conn.execute(
        """INSERT INTO evaluations
           (run_id, question, faithfulness, relevancy, completeness, rouge_l, overall_score,
            faithfulness_reason, relevancy_reason, completeness_reason,
            factual_anchor_score, factual_supported, factual_hallucinated,
            golden_rouge_l, contradicts_golden, contradiction_detail, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id, question,
            scores.get("faithfulness"),   scores.get("relevancy"),
            scores.get("completeness"),   scores.get("rouge_l"),
            scores.get("overall"),
            scores.get("faithfulness_reason", ""),
            scores.get("relevancy_reason",    ""),
            scores.get("completeness_reason", ""),
            scores.get("factual_anchor_score"),
            json.dumps(scores.get("factual_supported",    [])),
            json.dumps(scores.get("factual_hallucinated", [])),
            scores.get("golden_rouge_l"),
            int(scores.get("contradicts_golden", False)),
            scores.get("contradiction_detail", ""),
            datetime.utcnow().isoformat(),
        ),
    )
    conn.execute("UPDATE test_runs SET evaluated = 1 WHERE id = ?", (run_id,))
    conn.commit()
    conn.close()


def save_golden_answer(question: str, golden_answer: str, factual_anchors: str):
    conn = get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO golden_answers (question, golden_answer, factual_anchors, created_at)
           VALUES (?, ?, ?, ?)""",
        (question, golden_answer, factual_anchors, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_golden_answer(question: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM golden_answers WHERE question = ?", (question,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def save_consistency_report(question: str, data: dict):
    conn = get_conn()
    existing = conn.execute(
        "SELECT id FROM consistency_reports WHERE question = ?", (question,)
    ).fetchone()
    if existing:
        conn.execute(
            """UPDATE consistency_reports SET consistency_score=?, contradiction_rate=?,
               drift_score=?, total_runs=?, flagged=?, contradiction_details=?, timestamp=?
               WHERE question=?""",
            (data["consistency_score"], data["contradiction_rate"], data["drift_score"],
             data["total_runs"], data["flagged"], json.dumps(data.get("contradiction_details", [])),
             datetime.utcnow().isoformat(), question),
        )
    else:
        conn.execute(
            """INSERT INTO consistency_reports
               (question, consistency_score, contradiction_rate, drift_score, total_runs,
                flagged, contradiction_details, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (question, data["consistency_score"], data["contradiction_rate"],
             data["drift_score"], data["total_runs"], data["flagged"],
             json.dumps(data.get("contradiction_details", [])),
             datetime.utcnow().isoformat()),
        )
    conn.commit()
    conn.close()


def get_all_answers_for_question(question: str) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        """SELECT tr.id, tr.answer, tr.retrieved_context, tr.timestamp,
                  e.faithfulness, e.relevancy, e.completeness, e.rouge_l, e.overall_score
           FROM test_runs tr
           LEFT JOIN evaluations e ON tr.id = e.run_id
           WHERE tr.question = ?
           ORDER BY tr.timestamp ASC""",
        (question,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_evaluated_data() -> dict:
    conn = get_conn()
    evaluations = conn.execute(
        """SELECT e.*, tr.answer, tr.retrieved_context, tr.timestamp as run_time
           FROM evaluations e
           JOIN test_runs tr ON e.run_id = tr.id
           ORDER BY e.timestamp ASC"""
    ).fetchall()
    consistency = conn.execute(
        "SELECT * FROM consistency_reports ORDER BY consistency_score ASC"
    ).fetchall()
    conn.close()
    return {
        "evaluations": [dict(r) for r in evaluations],
        "consistency": [dict(r) for r in consistency],
    }


def save_generated_questions(questions: list[dict]):
    conn = get_conn()
    conn.execute("DELETE FROM generated_questions")
    for q in questions:
        conn.execute(
            "INSERT INTO generated_questions (question, category, created_at) VALUES (?, ?, ?)",
            (q["question"], q.get("category", "general"), datetime.utcnow().isoformat()),
        )
    conn.commit()
    conn.close()


def get_generated_questions() -> list[dict]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM generated_questions ORDER BY id ASC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_next_question_index(total: int) -> int:
    conn = get_conn()
    count = conn.execute("SELECT COUNT(*) FROM test_runs").fetchone()[0]
    conn.close()
    return count % total
