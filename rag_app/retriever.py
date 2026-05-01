from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


_vectorizer = None
_matrix     = None
_chunks     = []


def _build_chunks(documents: list[dict]) -> list[dict]:
    chunks = []
    for doc in documents:
        paragraphs = [p.strip() for p in doc["content"].split("\n\n") if p.strip()]
        for para in paragraphs:
            chunks.append({"source": doc["title"], "text": para})
    return chunks


def _build_index(documents: list[dict]):
    global _vectorizer, _matrix, _chunks
    _chunks     = _build_chunks(documents)
    texts       = [c["text"] for c in _chunks]
    _vectorizer = TfidfVectorizer(stop_words="english")
    _matrix     = _vectorizer.fit_transform(texts)


def reload():
    """Rebuild index from current document store (call after upload)."""
    from rag_app.document_store import get_all_documents
    _build_index(get_all_documents())


def _ensure_index():
    if _vectorizer is None:
        reload()


def retrieve(query: str, top_k: int = 4) -> list[dict]:
    _ensure_index()
    query_vec = _vectorizer.transform([query])
    scores    = cosine_similarity(query_vec, _matrix).flatten()
    top_idx   = np.argsort(scores)[::-1][:top_k]
    results   = []
    for idx in top_idx:
        if scores[idx] > 0:
            results.append({
                "source": _chunks[idx]["source"],
                "text":   _chunks[idx]["text"],
                "score":  float(scores[idx]),
            })
    return results
