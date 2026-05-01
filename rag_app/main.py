import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel
from rag_app.retriever import retrieve, reload
from rag_app.document_store import get_all_documents, get_uploaded_documents
import llm_client

app = FastAPI(title="Employee Policy RAG App")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_context: list[dict]
    sources: list[str]


@app.get("/")
def root():
    docs = get_all_documents()
    return {
        "app":         "Employee Policy RAG System",
        "description": "Answer questions about employee policies using RAG",
        "topics":      [doc["title"] for doc in docs],
        "endpoint":    "POST /query",
    }


@app.get("/topics")
def get_topics():
    docs = get_all_documents()
    return {
        "topics": [
            {"title": doc["title"], "summary": doc["content"].strip().split("\n")[0]}
            for doc in docs
        ]
    }


@app.get("/documents")
def list_documents():
    all_docs      = get_all_documents()
    uploaded_docs = get_uploaded_documents()
    uploaded_titles = {u["title"] for u in uploaded_docs}
    return {
        "total": len(all_docs),
        "documents": [
            {
                "title":    doc["title"],
                "type":     "uploaded" if doc["title"] in uploaded_titles else "default",
                "preview":  doc["content"].strip()[:120] + "...",
            }
            for doc in all_docs
        ],
    }


@app.post("/reload")
def reload_index():
    """Rebuild TF-IDF index after document changes."""
    reload()
    docs = get_all_documents()
    return {"status": "ok", "documents_loaded": len(docs)}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    chunks = retrieve(request.question, top_k=4)

    if not chunks:
        return QueryResponse(
            question=request.question,
            answer="I could not find relevant information in the policy documents.",
            retrieved_context=[],
            sources=[],
        )

    context_text = "\n\n".join(
        f"[{c['source']}]\n{c['text']}" for c in chunks
    )

    prompt = f"""You are an HR assistant helping employees understand company policies.
Answer the question below using ONLY the provided policy context.
If the answer is not in the context, say so clearly.

CONTEXT:
{context_text}

QUESTION: {request.question}

ANSWER:"""

    answer = llm_client.chat([{"role": "user", "content": prompt}], temperature=0.1)

    return QueryResponse(
        question=request.question,
        answer=answer,
        retrieved_context=chunks,
        sources=list({c["source"] for c in chunks}),
    )
