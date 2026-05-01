import sqlite3
import json
import os
from datetime import datetime, timezone

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

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
            eval_version TEXT,
            context_precision REAL,
            context_recall REAL,
            context_precision_reason TEXT,
            context_recall_reason TEXT,
            judge_count INTEGER DEFAULT 1,
            judge_disputed INTEGER DEFAULT 0,
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

        CREATE TABLE IF NOT EXISTS human_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            verdict TEXT,
            human_score REAL,
            notes TEXT,
            reviewed_by TEXT,
            reviewed_at TEXT,
            FOREIGN KEY (run_id) REFERENCES test_runs(id)
        );

        CREATE TABLE IF NOT EXISTS manual_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL UNIQUE,
            category TEXT DEFAULT 'manual',
            question_type TEXT DEFAULT 'standard',
            expected_answer TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            total_runs INTEGER,
            avg_overall REAL,
            avg_faithfulness REAL,
            avg_relevancy REAL,
            avg_completeness REAL,
            avg_factual REAL,
            avg_golden_rouge REAL,
            flagged_questions INTEGER,
            contradiction_count INTEGER,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS generated_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            category TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS llm_cache (
            cache_key  TEXT NOT NULL,
            metric     TEXT NOT NULL,
            result     TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (cache_key, metric)
        );

        CREATE TABLE IF NOT EXISTS dead_letter_queue (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            question     TEXT NOT NULL,
            error        TEXT,
            attempts     INTEGER DEFAULT 1,
            created_at   TEXT NOT NULL,
            last_attempt TEXT NOT NULL
        );
    """)
    conn.commit()

    # Migrate existing evaluations table - add new columns if missing
    new_cols = [
        ("factual_anchor_score",    "REAL"),
        ("factual_supported",       "TEXT"),
        ("factual_hallucinated",    "TEXT"),
        ("golden_rouge_l",          "REAL"),
        ("contradicts_golden",      "INTEGER DEFAULT 0"),
        ("contradiction_detail",    "TEXT"),
        ("eval_version",            "TEXT"),
        ("context_precision",       "REAL"),
        ("context_recall",          "REAL"),
        ("context_precision_reason","TEXT"),
        ("context_recall_reason",   "TEXT"),
        ("judge_count",             "INTEGER DEFAULT 1"),
        ("judge_disputed",          "INTEGER DEFAULT 0"),
    ]
    for col, col_type in new_cols:
        try:
            conn.execute(f"ALTER TABLE evaluations ADD COLUMN {col} {col_type}")
            conn.commit()
        except Exception:
            pass

    conn.close()


# -- Dead Letter Queue --------------------------------------------------------

def save_to_dlq(question: str, error: str):
    conn = get_conn()
    existing = conn.execute(
        "SELECT id, attempts FROM dead_letter_queue WHERE question = ?", (question,)
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE dead_letter_queue SET attempts = ?, error = ?, last_attempt = ? WHERE id = ?",
            (existing["attempts"] + 1, error, _now(), existing["id"]),
        )
    else:
        conn.execute(
            "INSERT INTO dead_letter_queue (question, error, attempts, created_at, last_attempt)"
            " VALUES (?,?,1,?,?)",
            (question, error, _now(), _now()),
        )
    conn.commit()
    conn.close()


def get_dlq_questions(max_attempts: int = 3) -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM dead_letter_queue WHERE attempts <= ? ORDER BY attempts ASC",
        (max_attempts,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def remove_from_dlq(question: str):
    conn = get_conn()
    conn.execute("DELETE FROM dead_letter_queue WHERE question = ?", (question,))
    conn.commit()
    conn.close()


# -- Question Prioritization --------------------------------------------------

def get_question_scores() -> dict:
    conn = get_conn()
    rows = conn.execute(
        "SELECT question, AVG(overall_score) as avg_score, COUNT(*) as run_count"
        " FROM evaluations GROUP BY question"
    ).fetchall()
    conn.close()
    return {r["question"]: {"avg_score": r["avg_score"], "run_count": r["run_count"]}
            for r in rows}


# -- Test Runs ----------------------------------------------------------------

def save_test_run(question: str, answer: str, retrieved_context: list, sources: list) -> int:
    conn = get_conn()
    cursor = conn.execute(
        "INSERT INTO test_runs (question, answer, retrieved_context, sources, timestamp)"
        " VALUES (?, ?, ?, ?, ?)",
        (question, answer, json.dumps(retrieved_context), json.dumps(sources), _now()),
    )
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return run_id


def get_unevaluated_runs() -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM test_runs WHERE evaluated = 0 ORDER BY timestamp ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# -- Evaluations --------------------------------------------------------------

def save_evaluation(run_id: int, question: str, scores: dict):
    conn = get_conn()
    conn.execute(
        "INSERT INTO evaluations"
        " (run_id, question, faithfulness, relevancy, completeness, rouge_l, overall_score,"
        "  faithfulness_reason, relevancy_reason, completeness_reason,"
        "  factual_anchor_score, factual_supported, factual_hallucinated,"
        "  golden_rouge_l, contradicts_golden, contradiction_detail,"
        "  eval_version, judge_count, judge_disputed,"
        "  context_precision, context_recall,"
        "  context_precision_reason, context_recall_reason,"
        "  timestamp)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id, question,
            scores.get("faithfulness"),     scores.get("relevancy"),
            scores.get("completeness"),     scores.get("rouge_l"),
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
            scores.get("eval_version", ""),
            scores.get("judge_count", 1),
            int(scores.get("judge_disputed", False)),
            scores.get("context_precision"),
            scores.get("context_recall"),
            scores.get("context_precision_reason", ""),
            scores.get("context_recall_reason", ""),
            _now(),
        ),
    )
    conn.execute("UPDATE test_runs SET evaluated = 1 WHERE id = ?", (run_id,))
    conn.commit()
    conn.close()


# -- Golden Answers -----------------------------------------------------------

def save_golden_answer(question: str, golden_answer: str, factual_anchors: str):
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO golden_answers (question, golden_answer, factual_anchors, created_at)"
        " VALUES (?, ?, ?, ?)",
        (question, golden_answer, factual_anchors, _now()),
    )
    conn.commit()
    conn.close()


def get_golden_answer(question: str):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM golden_answers WHERE question = ?", (question,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# -- Consistency Reports ------------------------------------------------------

def save_consistency_report(question: str, data: dict):
    conn = get_conn()
    existing = conn.execute(
        "SELECT id FROM consistency_reports WHERE question = ?", (question,)
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE consistency_reports SET consistency_score=?, contradiction_rate=?,"
            " drift_score=?, total_runs=?, flagged=?, contradiction_details=?, timestamp=?"
            " WHERE question=?",
            (data["consistency_score"], data["contradiction_rate"],
             data["drift_score"], data["total_runs"], data["flagged"],
             json.dumps(data.get("contradiction_details", [])),
             _now(), question),
        )
    else:
        conn.execute(
            "INSERT INTO consistency_reports"
            " (question, consistency_score, contradiction_rate, drift_score, total_runs,"
            "  flagged, contradiction_details, timestamp)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (question, data["consistency_score"], data["contradiction_rate"],
             data["drift_score"], data["total_runs"], data["flagged"],
             json.dumps(data.get("contradiction_details", [])), _now()),
        )
    conn.commit()
    conn.close()


def get_all_answers_for_question(question: str) -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT tr.id, tr.answer, tr.retrieved_context, tr.timestamp,"
        "       e.faithfulness, e.relevancy, e.completeness, e.rouge_l, e.overall_score"
        " FROM test_runs tr"
        " LEFT JOIN evaluations e ON tr.id = e.run_id"
        " WHERE tr.question = ?"
        " ORDER BY tr.timestamp ASC",
        (question,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_evaluated_data() -> dict:
    conn = get_conn()
    evaluations = conn.execute(
        "SELECT e.*, tr.answer, tr.retrieved_context, tr.timestamp as run_time"
        " FROM evaluations e"
        " JOIN test_runs tr ON e.run_id = tr.id"
        " ORDER BY e.timestamp ASC"
    ).fetchall()
    consistency = conn.execute(
        "SELECT * FROM consistency_reports ORDER BY consistency_score ASC"
    ).fetchall()
    conn.close()
    return {
        "evaluations": [dict(r) for r in evaluations],
        "consistency": [dict(r) for r in consistency],
    }


def save_generated_questions(questions: list):
    conn = get_conn()
    conn.execute("DELETE FROM generated_questions")
    for q in questions:
        conn.execute(
            "INSERT INTO generated_questions (question, category, created_at) VALUES (?, ?, ?)",
            (q["question"], q.get("category", "general"), _now()),
        )
    conn.commit()
    conn.close()


def get_generated_questions() -> list:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM generated_questions ORDER BY id ASC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_next_question_index(total: int) -> int:
    conn = get_conn()
    count = conn.execute("SELECT COUNT(*) FROM test_runs").fetchone()[0]
    conn.close()
    return count % total


# -- Human Review -------------------------------------------------------------

def get_runs_pending_review(score_threshold: float = 0.60) -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT tr.id, tr.question, tr.answer, tr.timestamp,"
        "       e.overall_score, e.faithfulness, e.factual_anchor_score"
        " FROM test_runs tr"
        " JOIN evaluations e ON tr.id = e.run_id"
        " WHERE e.overall_score < ?"
        "   AND tr.id NOT IN (SELECT run_id FROM human_reviews)"
        " ORDER BY e.overall_score ASC",
        (score_threshold,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_human_review(run_id: int, question: str, answer: str,
                      verdict: str, human_score: float, notes: str, reviewed_by: str):
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO human_reviews"
        " (run_id, question, answer, verdict, human_score, notes, reviewed_by, reviewed_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, question, answer, verdict, human_score, notes, reviewed_by, _now()),
    )
    conn.commit()
    conn.close()


def get_all_human_reviews() -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM human_reviews ORDER BY reviewed_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_golden_answer(question: str, verified_answer: str):
    conn = get_conn()
    conn.execute(
        "UPDATE golden_answers SET golden_answer = ? WHERE question = ?",
        (verified_answer, question),
    )
    conn.commit()
    conn.close()


# -- Manual Questions ---------------------------------------------------------

def save_manual_question(question: str, category: str,
                         question_type: str, expected_answer: str = ""):
    conn = get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO manual_questions"
        " (question, category, question_type, expected_answer, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (question, category, question_type, expected_answer, _now()),
    )
    conn.commit()
    conn.close()


def get_manual_questions() -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM manual_questions ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_manual_question(question_id: int):
    conn = get_conn()
    conn.execute("DELETE FROM manual_questions WHERE id = ?", (question_id,))
    conn.commit()
    conn.close()


# -- Snapshots / Regression Tracking -----------------------------------------

def take_snapshot(name: str) -> dict:
    conn = get_conn()

    def avg(col):
        row = conn.execute(
            f"SELECT AVG({col}) FROM evaluations WHERE {col} IS NOT NULL"
        ).fetchone()[0]
        return round(row, 4) if row else 0.0

    total_runs  = conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
    flagged     = conn.execute(
        "SELECT COUNT(*) FROM consistency_reports WHERE flagged = 1"
    ).fetchone()[0]
    contradicts = conn.execute(
        "SELECT COUNT(*) FROM evaluations WHERE contradicts_golden = 1"
    ).fetchone()[0]

    snap = {
        "name":                name,
        "total_runs":          total_runs,
        "avg_overall":         avg("overall_score"),
        "avg_faithfulness":    avg("faithfulness"),
        "avg_relevancy":       avg("relevancy"),
        "avg_completeness":    avg("completeness"),
        "avg_factual":         avg("factual_anchor_score"),
        "avg_golden_rouge":    avg("golden_rouge_l"),
        "flagged_questions":   flagged,
        "contradiction_count": contradicts,
        "created_at":          _now(),
    }

    conn.execute(
        "INSERT INTO snapshots"
        " (name, total_runs, avg_overall, avg_faithfulness, avg_relevancy,"
        "  avg_completeness, avg_factual, avg_golden_rouge,"
        "  flagged_questions, contradiction_count, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        tuple(snap.values()),
    )
    conn.commit()
    conn.close()
    return snap


def get_snapshots() -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM snapshots ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
