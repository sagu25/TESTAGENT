import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import asyncio
import random
import llm_client
import storage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

RAG_APP_URL = os.getenv("RAG_APP_URL", "http://localhost:8000")
QUESTIONS_PER_CYCLE = int(os.getenv("QUESTIONS_PER_CYCLE", "3"))


# ── Retry + Async HTTP ────────────────────────────────────────────────────────

async def _fire_async(session, question: str) -> dict | None:
    import aiohttp
    try:
        async with session.post(
            f"{RAG_APP_URL}/query",
            json={"question": question},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            return None
    except Exception as e:
        return {"_error": str(e)}


async def fire_questions_parallel(questions: list[str]) -> list[dict]:
    """
    Fire multiple questions to the RAG app concurrently.
    Returns list of {question, result} dicts.
    """
    import aiohttp
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = [_fire_async(session, q) for q in questions]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for question, response in zip(questions, responses):
            if isinstance(response, Exception):
                results.append({"question": question, "result": None,
                                 "error": str(response)})
            elif response and "_error" in response:
                results.append({"question": question, "result": None,
                                 "error": response["_error"]})
            else:
                results.append({"question": question, "result": response,
                                 "error": None})
    return results


# ── Smart Question Prioritization ─────────────────────────────────────────────

def get_prioritized_questions(all_questions: list[dict], n: int) -> list[str]:
    """
    Select n questions using weighted prioritization:
      - Never-asked questions:              weight 4.0 (highest)
      - Low-scoring questions (< 0.60):     weight 3.0
      - Flagged inconsistent questions:     weight 2.5
      - Medium-scoring (0.60 – 0.80):      weight 1.0
      - High-scoring questions (> 0.80):   weight 0.5
      - Dead letter queue (retry):         weight 3.5
    """
    past_scores  = storage.get_question_scores()
    dlq_items    = {d["question"] for d in storage.get_dlq_questions()}
    cons_data    = storage.get_all_evaluated_data()
    flagged      = {c["question"] for c in cons_data.get("consistency", [])
                    if c.get("flagged")}

    weighted = []
    for q_entry in all_questions:
        q = q_entry["question"] if isinstance(q_entry, dict) else q_entry
        info = past_scores.get(q)

        if q in dlq_items:
            weight = 3.5
        elif info is None:
            weight = 4.0  # never asked
        elif q in flagged:
            weight = 2.5  # inconsistent
        elif info["avg_score"] < 0.60:
            weight = 3.0  # poor performer
        elif info["avg_score"] < 0.80:
            weight = 1.0  # medium
        else:
            weight = 0.5  # good — test less often

        weighted.append((q, weight))

    if not weighted:
        return []

    questions_list = [q for q, _ in weighted]
    weights_list   = [w for _, w in weighted]
    k              = min(n, len(questions_list))
    selected       = random.choices(questions_list, weights=weights_list, k=k)
    # Deduplicate while preserving order
    seen, unique = set(), []
    for q in selected:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique


# ── Question Generation ───────────────────────────────────────────────────────

def analyze_app_and_generate_questions() -> list[dict]:
    import requests
    print("[TestAgent] Analyzing RAG app to generate questions...")
    try:
        info        = requests.get(f"{RAG_APP_URL}/", timeout=10).json()
        topics_resp = requests.get(f"{RAG_APP_URL}/topics", timeout=10).json()
        topics      = topics_resp.get("topics", [])
        description = info.get("description", "A RAG-based question answering system")
        topic_names = [t["title"] for t in topics]
    except Exception as e:
        print(f"[TestAgent] Could not fetch app info: {e}")
        description = "A RAG-based question answering system"
        topic_names = []

    topic_section = f"\nTopics covered: {', '.join(topic_names)}" if topic_names else ""

    prompt = f"""You are a QA engineer generating diverse test questions for a RAG system.

App: {description}{topic_section}

Generate exactly 15 test questions covering:
- Factual   (specific numbers, thresholds, dates)
- Procedural (steps, process, how-to)
- Conditional (edge cases, what-if scenarios)
- Eligibility (who qualifies, who is excluded)
- Adversarial (boundary cases, slightly wrong premise)

Make questions specific and realistic. Avoid trivial yes/no questions.

Respond ONLY with valid JSON array:
[
  {{"question": "...", "category": "factual"}},
  ...15 total
]"""

    response  = llm_client.chat([{"role": "user", "content": prompt}], temperature=0.7)
    import re
    match = re.search(r"\[.*\]", response, re.DOTALL)
    if match:
        try:
            questions = json.loads(match.group())
            print(f"[TestAgent] Generated {len(questions)} questions.")
            return questions
        except json.JSONDecodeError:
            pass
    return [{"question": "What topics does this system cover?", "category": "general"}]


def get_or_generate_questions() -> list[dict]:
    existing = storage.get_generated_questions()
    if not existing:
        existing = analyze_app_and_generate_questions()
        storage.save_generated_questions(existing)

    manual = storage.get_manual_questions()
    manual_formatted = [
        {"question": m["question"], "category": m.get("question_type", "manual")}
        for m in manual
    ]
    all_q = existing + manual_formatted
    print(f"[TestAgent] {len(existing)} auto + {len(manual_formatted)} manual = {len(all_q)} total questions")
    return all_q


# ── Dead Letter Queue Retry ───────────────────────────────────────────────────

def retry_dead_letter_queue():
    dlq = storage.get_dlq_questions(max_attempts=3)
    if not dlq:
        return
    print(f"[TestAgent] Retrying {len(dlq)} failed question(s) from DLQ...")
    import requests
    for item in dlq:
        q = item["question"]
        try:
            resp = requests.post(f"{RAG_APP_URL}/query",
                                 json={"question": q}, timeout=30)
            if resp.status_code == 200:
                data   = resp.json()
                run_id = storage.save_test_run(
                    question         = q,
                    answer           = data.get("answer", ""),
                    retrieved_context= data.get("retrieved_context", []),
                    sources          = data.get("sources", []),
                )
                storage.remove_from_dlq(q)
                print(f"[TestAgent] DLQ retry succeeded for: {q[:60]}")
            else:
                storage.save_to_dlq(q, f"HTTP {resp.status_code}")
        except Exception as e:
            storage.save_to_dlq(q, str(e))


# ── Main Run ──────────────────────────────────────────────────────────────────

def run():
    print(f"\n[TestAgent] Starting test run (parallel={QUESTIONS_PER_CYCLE} questions)...")

    # Retry any previously failed questions first
    retry_dead_letter_queue()

    all_questions = get_or_generate_questions()
    if not all_questions:
        print("[TestAgent] No questions available.")
        return

    # Smart prioritization
    selected = get_prioritized_questions(all_questions, n=QUESTIONS_PER_CYCLE)
    if not selected:
        selected = [all_questions[0]["question"] if isinstance(all_questions[0], dict)
                    else all_questions[0]]

    print(f"[TestAgent] Selected {len(selected)} questions to fire:")
    for q in selected:
        print(f"  → {q[:80]}")

    # Parallel firing via asyncio + aiohttp
    results = asyncio.run(fire_questions_parallel(selected))

    saved = 0
    for item in results:
        question = item["question"]
        result   = item["result"]
        error    = item["error"]

        if error or not result:
            print(f"[TestAgent] FAILED: {question[:60]} — {error}")
            storage.save_to_dlq(question, error or "No response")
            continue

        run_id = storage.save_test_run(
            question          = question,
            answer            = result.get("answer", ""),
            retrieved_context = result.get("retrieved_context", []),
            sources           = result.get("sources", []),
        )
        print(f"[TestAgent] Saved run {run_id}: {question[:60]}")
        saved += 1

    print(f"[TestAgent] {saved}/{len(selected)} questions saved successfully.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    storage.init_db()
    run()
