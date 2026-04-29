import json
import re
from rouge_score import rouge_scorer
import llm_client


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


def evaluate_faithfulness(question: str, answer: str, context: str) -> dict:
    prompt = f"""You are an expert evaluator assessing RAG system quality.

TASK: Evaluate if the answer is FAITHFUL to the retrieved context.
Faithfulness means: every factual claim in the answer must be supported by the context.
Penalize heavily if the answer contains facts NOT present in the context.

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

ANSWER: {answer}

Respond ONLY with valid JSON:
{{
  "score": <float 0.0 to 1.0>,
  "reason": "<one sentence explanation>",
  "unsupported_claims": ["<claim1>", "<claim2>"]
}}

Score guide: 1.0=fully faithful, 0.5=partially faithful, 0.0=contradicts or ignores context"""

    response = llm_client.chat([{"role": "user", "content": prompt}], temperature=0.0)
    parsed = _parse_json_from_llm(response)
    return {
        "score": float(parsed.get("score", 0.5)),
        "reason": parsed.get("reason", response[:200]),
    }


def evaluate_relevancy(question: str, answer: str) -> dict:
    prompt = f"""You are an expert evaluator assessing RAG system quality.

TASK: Evaluate if the answer is RELEVANT to the question.
Relevancy means: the answer directly addresses what was asked, without going off-topic.

QUESTION: {question}

ANSWER: {answer}

Respond ONLY with valid JSON:
{{
  "score": <float 0.0 to 1.0>,
  "reason": "<one sentence explanation>"
}}

Score guide: 1.0=directly answers the question, 0.5=partially relevant, 0.0=completely off-topic"""

    response = llm_client.chat([{"role": "user", "content": prompt}], temperature=0.0)
    parsed = _parse_json_from_llm(response)
    return {
        "score": float(parsed.get("score", 0.5)),
        "reason": parsed.get("reason", response[:200]),
    }


def evaluate_completeness(question: str, answer: str, context: str) -> dict:
    prompt = f"""You are an expert evaluator assessing RAG system quality.

TASK: Evaluate if the answer is COMPLETE — does it cover all relevant information from the context needed to fully answer the question?

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

ANSWER: {answer}

Respond ONLY with valid JSON:
{{
  "score": <float 0.0 to 1.0>,
  "reason": "<one sentence explanation>",
  "missing_info": ["<missing point 1>", "<missing point 2>"]
}}

Score guide: 1.0=fully complete, 0.5=partially complete, 0.0=major information missing"""

    response = llm_client.chat([{"role": "user", "content": prompt}], temperature=0.0)
    parsed = _parse_json_from_llm(response)
    return {
        "score": float(parsed.get("score", 0.5)),
        "reason": parsed.get("reason", response[:200]),
    }


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
        "semantic_similarity": float(parsed.get("semantic_similarity", 0.5)),
        "contradicts": bool(parsed.get("contradicts", False)),
        "contradiction_detail": parsed.get("contradiction_detail", ""),
    }


def compute_overall_score(faithfulness: float, relevancy: float,
                          completeness: float, rouge_l: float) -> float:
    return round(
        0.30 * faithfulness +
        0.25 * relevancy +
        0.25 * completeness +
        0.20 * rouge_l,
        4
    )


def compute_consistency_score(answers: list[str], question: str) -> dict:
    if len(answers) < 2:
        return {
            "consistency_score": 1.0,
            "contradiction_rate": 0.0,
            "drift_score": 0.0,
            "total_runs": len(answers),
            "flagged": False,
            "contradiction_details": [],
        }

    similarities = []
    contradictions = []
    contradiction_details = []

    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            result = check_consistency_pair(question, answers[i], answers[j])
            similarities.append(result["semantic_similarity"])
            if result["contradicts"]:
                contradictions.append((i, j))
                contradiction_details.append({
                    "run_a": i + 1,
                    "run_b": j + 1,
                    "detail": result["contradiction_detail"],
                })

    avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0
    contradiction_rate = len(contradictions) / len(similarities) if similarities else 0.0

    first_last = check_consistency_pair(question, answers[0], answers[-1])
    drift_score = round(1.0 - first_last["semantic_similarity"], 4)

    consistency_score = round(
        avg_similarity * (1 - contradiction_rate) * (1 - drift_score * 0.5),
        4
    )

    return {
        "consistency_score": consistency_score,
        "contradiction_rate": round(contradiction_rate, 4),
        "drift_score": drift_score,
        "total_runs": len(answers),
        "flagged": consistency_score < 0.75,
        "contradiction_details": contradiction_details,
    }
