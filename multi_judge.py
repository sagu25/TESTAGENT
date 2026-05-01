"""
Multi-judge consensus evaluation.
Calls two independent LLM judges, averages scores, flags disputes.

Judge 1: Primary provider from LLM_PROVIDER in .env
Judge 2: Best available secondary provider
"""
import os
import re
import json
import time
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

DISPUTE_THRESHOLD = 0.25  # flag if judges disagree by more than this


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {}


def _get_all_clients() -> list[dict]:
    """Return all configured LLM clients."""
    clients = []

    azure_key      = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    if azure_key and azure_key != "your_azure_api_key_here" and azure_endpoint:
        clients.append({
            "name":   "Azure OpenAI",
            "client": AzureOpenAI(
                api_key        = azure_key,
                azure_endpoint = azure_endpoint,
                api_version    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            ),
            "model": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
        })

    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key and groq_key != "your_groq_api_key_here":
        clients.append({
            "name":   "Groq",
            "client": OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1"),
            "model":  os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        })

    grok_key = os.getenv("GROK_API_KEY", "")
    if grok_key and grok_key != "your_grok_api_key_here":
        clients.append({
            "name":   "Grok",
            "client": OpenAI(api_key=grok_key, base_url="https://api.x.ai/v1"),
            "model":  os.getenv("GROK_MODEL", "grok-3"),
        })

    return clients


def _call_judge(client_info: dict, prompt: str, retries: int = 3) -> str:
    wait = 5
    for attempt in range(retries):
        try:
            resp = client_info["client"].chat.completions.create(
                model       = client_info["model"],
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                time.sleep(wait)
                wait *= 2
            else:
                return ""
    return ""


def multi_judge(prompt: str, score_key: str = "score") -> dict:
    """
    Call all available judges with the same prompt.
    Returns averaged score + per-judge breakdown + dispute flag.
    """
    clients = _get_all_clients()
    if not clients:
        return {"score": 0.5, "judges": [], "disputed": False, "avg_score": 0.5}

    results = []
    for c in clients[:2]:  # use max 2 judges to limit cost
        raw     = _call_judge(c, prompt)
        parsed  = _parse_json(raw)
        score   = float(parsed.get(score_key, 0.5))
        reason  = parsed.get("reason", raw[:150] if raw else "No response")
        results.append({
            "judge":  c["name"],
            "score":  score,
            "reason": reason,
            "raw":    parsed,
        })

    if not results:
        return {"score": 0.5, "judges": [], "disputed": False, "avg_score": 0.5}

    scores    = [r["score"] for r in results]
    avg_score = round(sum(scores) / len(scores), 4)
    disputed  = (max(scores) - min(scores)) > DISPUTE_THRESHOLD if len(scores) > 1 else False

    return {
        "score":     avg_score,
        "avg_score": avg_score,
        "judges":    results,
        "disputed":  disputed,
        "min_score": min(scores),
        "max_score": max(scores),
        "disagreement": round(max(scores) - min(scores), 4) if len(scores) > 1 else 0.0,
    }


def multi_judge_faithfulness(question: str, answer: str, context: str, golden_answer: str) -> dict:
    prompt = f"""You are an expert RAG evaluator.

TASK: Score FAITHFULNESS — does the app answer match the golden reference and stay grounded in the context?

QUESTION: {question}

RETRIEVED CONTEXT: {context[:1500]}

GOLDEN ANSWER (ground truth): {golden_answer[:800]}

APP ANSWER: {answer}

Respond ONLY with JSON:
{{
  "score": <0.0 to 1.0>,
  "contradicts_golden": <true or false>,
  "contradiction_detail": "<specific contradiction or none>",
  "reason": "<one sentence>"
}}"""

    result = multi_judge(prompt)
    # Extract contradiction info from the judge that gave lower score
    lowest_judge = min(result["judges"], key=lambda x: x["score"]) if result["judges"] else {}
    raw = lowest_judge.get("raw", {})

    return {
        "score":                result["avg_score"],
        "judges":               result["judges"],
        "disputed":             result["disputed"],
        "disagreement":         result["disagreement"],
        "contradicts_golden":   bool(raw.get("contradicts_golden", False)),
        "contradiction_detail": raw.get("contradiction_detail", ""),
        "reason":               f"[Consensus of {len(result['judges'])} judges] " +
                                (result["judges"][0]["reason"] if result["judges"] else ""),
    }


def multi_judge_relevancy(question: str, answer: str, golden_answer: str) -> dict:
    prompt = f"""You are an expert RAG evaluator.

TASK: Score RELEVANCY — does the answer directly address the question?

QUESTION: {question}
GOLDEN ANSWER (reference): {golden_answer[:600]}
APP ANSWER: {answer}

Respond ONLY with JSON:
{{
  "score": <0.0 to 1.0>,
  "reason": "<one sentence>"
}}"""

    result = multi_judge(prompt)
    return {
        "score":        result["avg_score"],
        "judges":       result["judges"],
        "disputed":     result["disputed"],
        "disagreement": result["disagreement"],
        "reason":       f"[{len(result['judges'])} judges] " +
                        (result["judges"][0]["reason"] if result["judges"] else ""),
    }


def multi_judge_completeness(question: str, answer: str, golden_answer: str) -> dict:
    prompt = f"""You are an expert RAG evaluator.

TASK: Score COMPLETENESS — does the answer cover all important details from the golden reference?

QUESTION: {question}
GOLDEN ANSWER (benchmark): {golden_answer[:600]}
APP ANSWER: {answer}

Respond ONLY with JSON:
{{
  "score": <0.0 to 1.0>,
  "missing_details": ["<detail1>", "<detail2>"],
  "reason": "<one sentence>"
}}"""

    result = multi_judge(prompt)
    lowest = min(result["judges"], key=lambda x: x["score"]) if result["judges"] else {}
    return {
        "score":           result["avg_score"],
        "judges":          result["judges"],
        "disputed":        result["disputed"],
        "disagreement":    result["disagreement"],
        "missing_details": lowest.get("raw", {}).get("missing_details", []),
        "reason":          f"[{len(result['judges'])} judges] " +
                           (result["judges"][0]["reason"] if result["judges"] else ""),
    }
