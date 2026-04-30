import json
import re
from rouge_score import rouge_scorer
import llm_client
from factual_extractor import check_factual_anchors


def rouge_l_score(answer: str, reference: str) -> float:
    if not answer or not reference:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    result = scorer.score(reference, answer)
    return round(result["rougeL"].fmeasure, 4)


def _parse_json_from_llm(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


# ── Layer 1: Factual Anchor Check (pure code, zero LLM) ──────────────────────

def evaluate_factual_anchors(answer: str, source_context: str) -> dict:
    return check_factual_anchors(answer, source_context)


# ── Layer 2: Golden Answer ROUGE-L (pure math, zero LLM) ─────────────────────

def evaluate_golden_rouge_l(answer: str, golden_answer: str) -> float:
    return rouge_l_score(answer, golden_answer)


# ── Layer 3: Grounded LLM Judge (uses golden answer as reference) ─────────────

def evaluate_faithfulness_grounded(question: str, answer: str,
                                   context: str, golden_answer: str) -> dict:
    prompt = f"""You are an expert evaluator with access to both the retrieved context
AND a verified golden reference answer generated from the FULL policy documents.

TASK: Evaluate if the app's answer is FAITHFUL and ACCURATE.

QUESTION: {question}

RETRIEVED CONTEXT (what the RAG app used):
{context}

VERIFIED GOLDEN ANSWER (from full documents — treat as ground truth):
{golden_answer}

APP ANSWER (what we are evaluating):
{answer}

Check:
1. Does the app's answer contradict the golden answer on any specific fact?
2. Does the app's answer contain numbers or claims NOT present in the retrieved context?
3. Overall faithfulness to the source.

Respond ONLY with valid JSON:
{{
  "score": <float 0.0 to 1.0>,
  "contradicts_golden": <true or false>,
  "contradiction_detail": "<specific contradiction or 'none'>",
  "reason": "<one sentence>"
}}

Score: 1.0=fully faithful and matches golden, 0.5=partially faithful, 0.0=contradicts golden"""

    response = llm_client.chat([{"role": "user", "content": prompt}], temperature=0.0)
    parsed = _parse_json_from_llm(response)
    return {
        "score":                float(parsed.get("score", 0.5)),
        "contradicts_golden":   bool(parsed.get("contradicts_golden", False)),
        "contradiction_detail": parsed.get("contradiction_detail", ""),
        "reason":               parsed.get("reason", response[:200]),
    }


def evaluate_relevancy_grounded(question: str, answer: str, golden_answer: str) -> dict:
    prompt = f"""You are an expert evaluator.

TASK: Evaluate if the app's answer is RELEVANT to the question.
Use the golden answer to understand what a complete, relevant answer looks like.

QUESTION: {question}

GOLDEN ANSWER (reference): {golden_answer}

APP ANSWER: {answer}

Respond ONLY with valid JSON:
{{
  "score": <float 0.0 to 1.0>,
  "reason": "<one sentence>"
}}

Score: 1.0=fully relevant (matches golden's scope), 0.5=partially, 0.0=off-topic"""

    response = llm_client.chat([{"role": "user", "content": prompt}], temperature=0.0)
    parsed = _parse_json_from_llm(response)
    return {
        "score":  float(parsed.get("score", 0.5)),
        "reason": parsed.get("reason", response[:200]),
    }


def evaluate_completeness_grounded(question: str, answer: str, golden_answer: str) -> dict:
    prompt = f"""You are an expert evaluator.

TASK: Evaluate if the app's answer is COMPLETE compared to the golden reference answer.
The golden answer was generated from the FULL document — it is the benchmark for completeness.

QUESTION: {question}

GOLDEN ANSWER (benchmark): {golden_answer}

APP ANSWER: {answer}

What key facts or details are in the golden answer but MISSING from the app's answer?

Respond ONLY with valid JSON:
{{
  "score": <float 0.0 to 1.0>,
  "missing_details": ["<detail1>", "<detail2>"],
  "reason": "<one sentence>"
}}

Score: 1.0=covers everything in golden, 0.5=missing some details, 0.0=major info missing"""

    response = llm_client.chat([{"role": "user", "content": prompt}], temperature=0.0)
    parsed = _parse_json_from_llm(response)
    return {
        "score":           float(parsed.get("score", 0.5)),
        "missing_details": parsed.get("missing_details", []),
        "reason":          parsed.get("reason", response[:200]),
    }


# ── Overall Score (4-layer formula) ──────────────────────────────────────────

def compute_overall_score(faithfulness: float, relevancy: float, completeness: float,
                          rouge_l: float, factual_anchor_score: float = None,
                          golden_rouge_l: float = None) -> float:
    """
    Layer 1 (Factual Anchors) — 0.25 weight  → pure code, most reliable
    Layer 2 (Golden ROUGE-L)  — 0.25 weight  → math vs ground truth
    Layer 3 (LLM Faithfulness)— 0.25 weight  → grounded LLM judge
    Layer 3 (LLM Relevancy)   — 0.15 weight  → grounded LLM judge
    Layer 3 (LLM Completeness)— 0.10 weight  → grounded LLM judge
    """
    if factual_anchor_score is not None and golden_rouge_l is not None:
        score = (
            0.25 * factual_anchor_score +
            0.25 * golden_rouge_l       +
            0.25 * faithfulness         +
            0.15 * relevancy            +
            0.10 * completeness
        )
        # Hard penalty: if factual anchors fail badly, cap the score
        if factual_anchor_score < 0.3:
            score = min(score, 0.45)
    else:
        # Fallback to original formula if golden answer not available
        score = (
            0.30 * faithfulness +
            0.25 * relevancy    +
            0.25 * completeness +
            0.20 * rouge_l
        )
    return round(score, 4)


# ── Consistency Check (unchanged) ────────────────────────────────────────────

def check_consistency_pair(question: str, answer_a: str, answer_b: str) -> dict:
    prompt = f"""You are an expert evaluator detecting inconsistencies in RAG system responses.

TASK: Compare two answers to the same question and determine:
1. Are they semantically similar (same meaning, different words)?
2. Do they CONTRADICT each other on any factual claims?

QUESTION: {question}

ANSWER A: {answer_a}

ANSWER B: {answer_b}

Respond ONLY with valid JSON:
{{
  "semantic_similarity": <float 0.0 to 1.0>,
  "contradicts": <true or false>,
  "contradiction_detail": "<what specifically contradicts, or 'none'>"
}}

Semantic similarity: 1.0=identical meaning, 0.5=partially similar, 0.0=completely different"""

    response = llm_client.chat([{"role": "user", "content": prompt}], temperature=0.0)
    parsed = _parse_json_from_llm(response)
    return {
        "semantic_similarity":  float(parsed.get("semantic_similarity", 0.5)),
        "contradicts":          bool(parsed.get("contradicts", False)),
        "contradiction_detail": parsed.get("contradiction_detail", ""),
    }


def compute_consistency_score(answers: list[str], question: str) -> dict:
    if len(answers) < 2:
        return {
            "consistency_score":    1.0,
            "contradiction_rate":   0.0,
            "drift_score":          0.0,
            "total_runs":           len(answers),
            "flagged":              False,
            "contradiction_details": [],
        }

    similarities           = []
    contradictions         = []
    contradiction_details  = []

    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            result = check_consistency_pair(question, answers[i], answers[j])
            similarities.append(result["semantic_similarity"])
            if result["contradicts"]:
                contradictions.append((i, j))
                contradiction_details.append({
                    "run_a":  i + 1,
                    "run_b":  j + 1,
                    "detail": result["contradiction_detail"],
                })

    avg_similarity     = sum(similarities) / len(similarities) if similarities else 1.0
    contradiction_rate = len(contradictions) / len(similarities) if similarities else 0.0
    first_last         = check_consistency_pair(question, answers[0], answers[-1])
    drift_score        = round(1.0 - first_last["semantic_similarity"], 4)

    consistency_score = round(
        avg_similarity * (1 - contradiction_rate) * (1 - drift_score * 0.5),
        4
    )

    return {
        "consistency_score":    consistency_score,
        "contradiction_rate":   round(contradiction_rate, 4),
        "drift_score":          drift_score,
        "total_runs":           len(answers),
        "flagged":              consistency_score < 0.75,
        "contradiction_details": contradiction_details,
    }
