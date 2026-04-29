from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rag_app.documents import POLICY_DOCUMENTS


def _build_chunks():
    chunks = []
    for doc in POLICY_DOCUMENTS:
        paragraphs = [p.strip() for p in doc["content"].split("\n\n") if p.strip()]
        for para in paragraphs:
            chunks.append({"source": doc["title"], "text": para})
    return chunks


CHUNKS = _build_chunks()
CHUNK_TEXTS = [c["text"] for c in CHUNKS]

_vectorizer = TfidfVectorizer(stop_words="english")
_matrix = _vectorizer.fit_transform(CHUNK_TEXTS)


def retrieve(query: str, top_k: int = 4) -> list[dict]:
    query_vec = _vectorizer.transform([query])
    scores = cosine_similarity(query_vec, _matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "source": CHUNKS[idx]["source"],
                "text": CHUNKS[idx]["text"],
                "score": float(scores[idx]),
            })
    return results
