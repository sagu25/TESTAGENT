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


def _get_top_docs(question: str, top_k: int = 2) -> list[dict]:
    """Return the top_k most relevant full documents for the question using TF-IDF."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    doc_texts = [doc["content"] for doc in POLICY_DOCUMENTS]
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform(doc_texts)
    q_vec = vec.transform([question])
    scores = cosine_similarity(q_vec, matrix).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [POLICY_DOCUMENTS[i] for i in top_idx if scores[i] > 0]


def generate_golden_answer(question: str) -> dict:
    """
    Generate a ground-truth answer using the top-2 most relevant
    FULL policy documents — no retrieval truncation, no TF-IDF chunk gaps.
    This becomes the reference answer the evaluator judges against.
    """
    if not _DOCS_AVAILABLE or not POLICY_DOCUMENTS:
        return {"golden_answer": "", "factual_anchors": "{}"}

    top_docs   = _get_top_docs(question, top_k=2)
    full_context = "\n\n" + ("=" * 50 + "\n").join(
        f"DOCUMENT: {doc['title']}\n{doc['content'].strip()}"
        for doc in top_docs
    )

    prompt = f"""You are an authoritative HR expert with access to the COMPLETE employee policy documentation.

Your task is to produce a VERIFIED REFERENCE ANSWER for the question below.
Rules:
- Use ONLY information from the provided documents.
- Include exact numbers, percentages, time periods, and dollar amounts as written.
- Be complete — cover every relevant detail from the documents.
- If the documents do not contain an answer, say so explicitly.

POLICY DOCUMENTS:
{full_context}

QUESTION: {question}

VERIFIED REFERENCE ANSWER:"""

    answer = llm_client.chat([{"role": "user", "content": prompt}], temperature=0.0)
    anchors = extract_factual_anchors(answer)

    return {
        "golden_answer":   answer,
        "factual_anchors": json.dumps(anchors),
    }


def get_or_generate(question: str) -> dict | None:
    existing = storage.get_golden_answer(question)
    if existing:
        return existing

    if not _DOCS_AVAILABLE:
        print(f"[GoldenGen] Source documents not accessible. Skipping golden answer.")
        return None

    print(f"[GoldenGen] Generating golden answer for: {question[:70]}...")
    result = generate_golden_answer(question)
    if result["golden_answer"]:
        storage.save_golden_answer(question, result["golden_answer"], result["factual_anchors"])
        print(f"[GoldenGen] Saved.")
        return storage.get_golden_answer(question)

    return None
