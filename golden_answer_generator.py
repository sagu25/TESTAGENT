import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import llm_client
import storage
from factual_extractor import extract_factual_anchors

try:
    from rag_app.documents import POLICY_DOCUMENTS
    _DOCS_AVAILABLE = True
except ImportError:
    _DOCS_AVAILABLE = False
    POLICY_DOCUMENTS = []


def _is_blueverse_mode() -> bool:
    return os.getenv("RAG_APP_URL", "").lower() == "blueverse"


def _get_top_docs(question: str, top_k: int = 2) -> list[dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    doc_texts = [doc["content"] for doc in POLICY_DOCUMENTS]
    vec    = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform(doc_texts)
    q_vec  = vec.transform([question])
    scores = cosine_similarity(q_vec, matrix).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [POLICY_DOCUMENTS[i] for i in top_idx if scores[i] > 0]


def generate_golden_answer_from_blueverse(question: str) -> dict:
    """
    In Blueverse mode: generate golden answer by asking Blueverse itself
    with a comprehensive 'give me the full authoritative answer' prompt.
    This ensures the golden reference matches what Blueverse knows.
    """
    import blueverse_connector

    REQUEST_FIELD  = os.getenv("BLUEVERSE_REQUEST_FIELD", "query")
    AGENT_ID       = os.getenv("BLUEVERSE_AGENT_ID", "")
    AGENT_NAME     = os.getenv("BLUEVERSE_AGENT_NAME", "")

    comprehensive_question = (
        f"Please provide a complete, detailed and authoritative answer to this question. "
        f"Include all relevant rules, numbers, limits, percentages and conditions: {question}"
    )

    print(f"[GoldenGen] Asking Blueverse for comprehensive golden answer...")
    result = blueverse_connector.query(comprehensive_question)

    if result and result.get("answer"):
        answer  = result["answer"]
        anchors = extract_factual_anchors(answer)
        return {
            "golden_answer":   answer,
            "factual_anchors": json.dumps(anchors),
        }

    # Fallback: use LLM with probe content as context
    probe_content = os.getenv("BLUEVERSE_DESCRIPTION", "")
    if probe_content:
        prompt = f"""You are an expert assistant. Based on the knowledge description below,
provide a complete reference answer for the question.

KNOWLEDGE BASE DESCRIPTION:
{probe_content}

QUESTION: {question}

COMPLETE REFERENCE ANSWER:"""
        answer  = llm_client.chat([{"role": "user", "content": prompt}], temperature=0.0)
        anchors = extract_factual_anchors(answer)
        return {"golden_answer": answer, "factual_anchors": json.dumps(anchors)}

    return {"golden_answer": "", "factual_anchors": "{}"}


def generate_golden_answer(question: str) -> dict:
    """
    Generate ground-truth reference answer.

    Blueverse mode: asks Blueverse itself for a comprehensive answer
                    so the reference matches what the agent actually knows
    Local RAG mode: uses top-2 most relevant full policy documents
    """
    if _is_blueverse_mode():
        return generate_golden_answer_from_blueverse(question)

    if not _DOCS_AVAILABLE or not POLICY_DOCUMENTS:
        return {"golden_answer": "", "factual_anchors": "{}"}

    top_docs     = _get_top_docs(question, top_k=2)
    full_context = "\n\n" + ("=" * 50 + "\n").join(
        f"DOCUMENT: {doc['title']}\n{doc['content'].strip()}"
        for doc in top_docs
    )

    prompt = f"""You are an authoritative HR expert with access to the COMPLETE employee policy documentation.

Produce a VERIFIED REFERENCE ANSWER for the question below.
Rules:
- Use ONLY information from the provided documents.
- Include exact numbers, percentages, time periods, and dollar amounts as written.
- Be complete — cover every relevant detail.
- If the documents do not contain an answer, say so explicitly.

POLICY DOCUMENTS:
{full_context}

QUESTION: {question}

VERIFIED REFERENCE ANSWER:"""

    answer  = llm_client.chat([{"role": "user", "content": prompt}], temperature=0.0)
    anchors = extract_factual_anchors(answer)

    return {
        "golden_answer":   answer,
        "factual_anchors": json.dumps(anchors),
    }


def get_or_generate(question: str) -> dict | None:
    existing = storage.get_golden_answer(question)
    if existing:
        return existing

    if not _is_blueverse_mode() and not _DOCS_AVAILABLE:
        print(f"[GoldenGen] Source documents not accessible. Skipping golden answer.")
        return None

    print(f"[GoldenGen] Generating golden answer for: {question[:70]}...")
    result = generate_golden_answer(question)
    if result["golden_answer"]:
        storage.save_golden_answer(question, result["golden_answer"], result["factual_anchors"])
        print(f"[GoldenGen] Saved.")
        return storage.get_golden_answer(question)

    return None
