"""
Retrieval Quality Metrics — Layer 0 (before generation).

Context Precision: Of retrieved chunks, what % are actually relevant?
Context Recall:    Does the retrieved set contain all info needed to answer?

These tell you WHERE a bad answer came from:
  Low precision + low recall → retrieval is broken (wrong chunks)
  High precision + high recall + low faithfulness → LLM generation is broken
"""
import json
import re
import llm_client


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {}


def evaluate_context_precision(question: str, chunks: list[dict]) -> dict:
    """
    Score each retrieved chunk: is it relevant to answering this question?
    Precision = relevant_chunks / total_chunks
    """
    if not chunks:
        return {"score": 0.0, "relevant_count": 0, "total_chunks": 0,
                "chunk_scores": [], "reason": "No chunks retrieved"}

    chunk_scores = []
    for i, chunk in enumerate(chunks):
        prompt = f"""{__import__('eval_version').PROMPT_CONTEXT_PRECISION}

QUESTION: {question}

RETRIEVED CHUNK [{i+1}]:
Source: {chunk.get('source', 'unknown')}
Text: {chunk.get('text', '')[:500]}

Is this chunk relevant and useful for answering the question?

Respond ONLY with JSON:
{{
  "relevant": <true or false>,
  "score": <0.0 to 1.0>,
  "reason": "<one sentence>"
}}"""
        response = llm_client.chat([{"role": "user", "content": prompt}], temperature=0.0)
        parsed   = _parse_json(response)
        chunk_scores.append({
            "chunk_index": i + 1,
            "source":      chunk.get("source", ""),
            "relevant":    bool(parsed.get("relevant", False)),
            "score":       float(parsed.get("score", 0.5)),
            "reason":      parsed.get("reason", ""),
        })

    relevant_count = sum(1 for c in chunk_scores if c["relevant"])
    precision      = round(relevant_count / len(chunks), 4)

    return {
        "score":          precision,
        "relevant_count": relevant_count,
        "total_chunks":   len(chunks),
        "chunk_scores":   chunk_scores,
        "reason":         f"{relevant_count}/{len(chunks)} retrieved chunks are relevant to the question",
    }


def evaluate_context_recall(question: str, golden_answer: str, chunks: list[dict]) -> dict:
    """
    Can the golden answer be derived from the retrieved chunks?
    If not — retrieval missed critical information.
    """
    if not chunks or not golden_answer:
        return {"score": 0.0, "reason": "No chunks or golden answer available",
                "missing_info": []}

    context_text = "\n\n".join(
        f"[{c.get('source','doc')}] {c.get('text','')[:400]}" for c in chunks
    )

    prompt = f"""{__import__('eval_version').PROMPT_CONTEXT_RECALL}

QUESTION: {question}

RETRIEVED CONTEXT (all chunks combined):
{context_text}

GOLDEN ANSWER (ground truth — what a complete answer should contain):
{golden_answer[:600]}

Can the golden answer be fully derived from the retrieved context?
What important information from the golden answer is MISSING from the retrieved context?

Respond ONLY with JSON:
{{
  "score": <0.0 to 1.0>,
  "missing_info": ["<missing fact 1>", "<missing fact 2>"],
  "reason": "<one sentence>"
}}

Score: 1.0 = context contains everything needed, 0.0 = context is completely missing key information"""

    response = llm_client.chat([{"role": "user", "content": prompt}], temperature=0.0)
    parsed   = _parse_json(response)

    return {
        "score":        float(parsed.get("score", 0.5)),
        "missing_info": parsed.get("missing_info", []),
        "reason":       parsed.get("reason", response[:200]),
    }
