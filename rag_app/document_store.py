"""
Dynamic document store.
Merges default policy docs with any user-uploaded documents.
Rebuilds the TF-IDF index automatically when documents change.
"""
import os
import json
import hashlib
from rag_app.documents import POLICY_DOCUMENTS

UPLOADS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
UPLOADS_META = os.path.join(UPLOADS_DIR, "meta.json")

os.makedirs(UPLOADS_DIR, exist_ok=True)


def _load_meta() -> list[dict]:
    if os.path.exists(UPLOADS_META):
        with open(UPLOADS_META, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_meta(meta: list[dict]):
    with open(UPLOADS_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def get_all_documents() -> list[dict]:
    """Return default docs + all uploaded docs."""
    docs = list(POLICY_DOCUMENTS)
    for entry in _load_meta():
        path = os.path.join(UPLOADS_DIR, entry["filename"])
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            docs.append({"title": entry["title"], "content": content})
    return docs


def add_document(title: str, content: str, original_filename: str) -> dict:
    """Save an uploaded document and register it."""
    file_id = hashlib.md5(content.encode()).hexdigest()[:8]
    safe_name = f"{file_id}_{original_filename.replace(' ', '_')}.txt"
    path = os.path.join(UPLOADS_DIR, safe_name)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    meta = _load_meta()
    # Remove old entry with same title if exists
    meta = [m for m in meta if m["title"] != title]
    meta.append({"title": title, "filename": safe_name, "original": original_filename})
    _save_meta(meta)
    return {"title": title, "filename": safe_name}


def remove_document(title: str):
    meta = _load_meta()
    to_remove = [m for m in meta if m["title"] == title]
    for entry in to_remove:
        path = os.path.join(UPLOADS_DIR, entry["filename"])
        if os.path.exists(path):
            os.remove(path)
    meta = [m for m in meta if m["title"] != title]
    _save_meta(meta)


def get_uploaded_documents() -> list[dict]:
    return _load_meta()


def clear_generated_questions():
    """Call this after document changes so test agent regenerates questions."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        import storage
        conn = storage.get_conn()
        conn.execute("DELETE FROM generated_questions")
        conn.execute("DELETE FROM golden_answers")
        conn.commit()
        conn.close()
    except Exception:
        pass
