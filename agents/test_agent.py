import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import requests
import llm_client
import storage


RAG_APP_URL = os.getenv("RAG_APP_URL", "http://localhost:8000")


def analyze_app_and_generate_questions() -> list[dict]:
    print("[TestAgent] Analyzing RAG app to generate questions...")

    try:
        info = requests.get(f"{RAG_APP_URL}/", timeout=10).json()
        topics_resp = requests.get(f"{RAG_APP_URL}/topics", timeout=10).json()
        topics = topics_resp.get("topics", [])
        app_description = info.get("description", "A RAG-based question answering system")
        topic_names = [t["title"] for t in topics]
    except Exception as e:
        print(f"[TestAgent] Could not fetch app info: {e}. Using generic approach.")
        app_description = "A RAG-based question answering system"
        topic_names = []

    topic_section = ""
    if topic_names:
        topic_section = f"\nThe app covers these topics: {', '.join(topic_names)}"

    prompt = f"""You are a QA engineer generating diverse test questions for a RAG system.

App description: {app_description}{topic_section}

Generate exactly 15 test questions to thoroughly evaluate this RAG system.
Cover these question types:
- Factual (specific numbers, dates, thresholds)
- Procedural (how to do something, steps involved)
- Conditional (what happens if X, edge cases)
- Eligibility (who qualifies for what)
- Comparative (difference between A and B)

Make questions realistic, specific, and varied. Do not make them trivial yes/no questions.

Respond ONLY with valid JSON array:
[
  {{"question": "...", "category": "factual"}},
  {{"question": "...", "category": "procedural"}},
  ...
]"""

    response = llm_client.chat([{"role": "user", "content": prompt}], temperature=0.7)

    import re
    match = re.search(r"\[.*\]", response, re.DOTALL)
    if match:
        try:
            questions = json.loads(match.group())
            print(f"[TestAgent] Generated {len(questions)} questions from app analysis.")
            return questions
        except json.JSONDecodeError:
            pass

    print("[TestAgent] Failed to parse generated questions. Using fallback probe.")
    return [{"question": "What topics does this system cover?", "category": "general"}]


def get_or_generate_questions() -> list[dict]:
    existing = storage.get_generated_questions()
    if not existing:
        existing = analyze_app_and_generate_questions()
        storage.save_generated_questions(existing)

    # Merge with manually added questions
    manual = storage.get_manual_questions()
    manual_formatted = [
        {"question": m["question"], "category": m.get("question_type", "manual")}
        for m in manual
    ]

    all_questions = existing + manual_formatted
    auto_count   = len(existing)
    manual_count = len(manual_formatted)
    print(f"[TestAgent] Questions: {auto_count} auto-generated + {manual_count} manual = {len(all_questions)} total")
    return all_questions


def fire_question(question: str) -> dict | None:
    try:
        response = requests.post(
            f"{RAG_APP_URL}/query",
            json={"question": question},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[TestAgent] Error calling RAG app: {e}")
        return None


def run():
    print("\n[TestAgent] Starting test run...")

    questions = get_or_generate_questions()
    if not questions:
        print("[TestAgent] No questions available. Skipping.")
        return

    idx = storage.get_next_question_index(len(questions))
    q_entry = questions[idx]
    question = q_entry["question"] if isinstance(q_entry, dict) else q_entry

    print(f"[TestAgent] Firing question [{idx + 1}/{len(questions)}]: {question}")

    result = fire_question(question)
    if not result:
        print("[TestAgent] No response from RAG app.")
        return

    run_id = storage.save_test_run(
        question=question,
        answer=result.get("answer", ""),
        retrieved_context=result.get("retrieved_context", []),
        sources=result.get("sources", []),
    )

    print(f"[TestAgent] Saved run ID {run_id}. Answer: {result.get('answer', '')[:100]}...")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    storage.init_db()
    run()
