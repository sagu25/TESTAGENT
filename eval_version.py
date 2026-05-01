"""
Evaluation Versioning.

Hashes the prompt templates + scoring weights → short version string.
Stored with every evaluation record so scores from different
evaluation logic are never compared against each other.

When you change a prompt or weight → new version → old scores tagged differently.
"""
import hashlib
import json

# These are the canonical prompt templates. Any change here = new version.
PROMPT_FAITHFULNESS = """You are an expert RAG evaluator with access to both the retrieved context
AND a verified golden reference answer generated from the FULL policy documents.
TASK: Evaluate if the app's answer is FAITHFUL and ACCURATE."""

PROMPT_RELEVANCY = """You are an expert evaluator.
TASK: Evaluate if the app's answer is RELEVANT to the question.
Use the golden answer to understand what a complete, relevant answer looks like."""

PROMPT_COMPLETENESS = """You are an expert evaluator.
TASK: Evaluate if the app's answer is COMPLETE compared to the golden reference answer.
The golden answer was generated from the FULL document — it is the benchmark for completeness."""

PROMPT_CONTEXT_PRECISION = """You are an expert RAG evaluator.
TASK: Evaluate if this retrieved context chunk is RELEVANT to the question."""

PROMPT_CONTEXT_RECALL = """You are an expert RAG evaluator.
TASK: Evaluate if the retrieved context contains enough information to answer the question fully."""

WEIGHTS = {
    "factual_anchor": 0.25,
    "golden_rouge_l": 0.25,
    "faithfulness":   0.25,
    "relevancy":      0.15,
    "completeness":   0.10,
}

_VERSION: str | None = None


def get_version() -> str:
    global _VERSION
    if _VERSION:
        return _VERSION
    content = json.dumps({
        "faithfulness":       PROMPT_FAITHFULNESS[:100],
        "relevancy":          PROMPT_RELEVANCY[:100],
        "completeness":       PROMPT_COMPLETENESS[:100],
        "context_precision":  PROMPT_CONTEXT_PRECISION[:100],
        "context_recall":     PROMPT_CONTEXT_RECALL[:100],
        "weights":            WEIGHTS,
    }, sort_keys=True)
    _VERSION = hashlib.md5(content.encode()).hexdigest()[:8]
    return _VERSION
