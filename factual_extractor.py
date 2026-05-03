import re


def extract_factual_anchors(text: str) -> dict:
    dollar_amounts = re.findall(r'\$[\d,]+(?:\.\d+)?(?:\s*(?:per\s+\w+|annually|monthly))?', text, re.IGNORECASE)
    percentages    = re.findall(r'\d+(?:\.\d+)?\s*%', text)
    time_facts     = re.findall(r'\d+\s*(?:days?|weeks?|months?|years?|hours?)\b', text, re.IGNORECASE)
    numbers        = re.findall(r'\b\d+\b', text)

    return {
        "dollars":     list(set(dollar_amounts)),
        "percentages": list(set(percentages)),
        "time_facts":  list(set(t.lower().strip() for t in time_facts)),
        "numbers":     list(set(numbers)),
    }


def check_factual_anchors(answer: str, source_context: str,
                           question: str = "") -> dict:
    """
    Check if facts in the answer are grounded in the source context.

    Numbers/facts that already appear in the QUESTION are excluded
    from the hallucination check — the app is allowed to repeat
    what was asked without being penalised.

    Example:
      Question: "If a flight is 7 hours long..."
      Answer:   "For a 7-hour flight, business class is permitted"
      Source:   "flights over 6 hours..."
      → "7" came from the question → NOT counted as hallucination
    """
    source  = extract_factual_anchors(source_context)
    ans     = extract_factual_anchors(answer)
    q_facts = extract_factual_anchors(question) if question else {
        "dollars": [], "percentages": [], "time_facts": [], "numbers": []
    }

    supported    = []
    hallucinated = []

    def _from_question(item: str, q_list: list) -> bool:
        item_clean = item.lower().strip()
        return any(item_clean in q.lower() or q.lower() in item_clean
                   for q in q_list)

    def _check(ans_list, src_list, q_list, label):
        for item in ans_list:
            if _from_question(item, q_list):
                supported.append(f"{label}:{item}(from-question)")
                continue
            item_clean = item.lower().strip()
            matched = any(item_clean in s.lower() or s.lower() in item_clean
                          for s in src_list)
            if matched:
                supported.append(f"{label}:{item}")
            else:
                hallucinated.append(f"{label}:{item}")

    _check(ans["dollars"],     source["dollars"],     q_facts["dollars"],     "dollar")
    _check(ans["percentages"], source["percentages"], q_facts["percentages"], "pct")
    _check(ans["time_facts"],  source["time_facts"],  q_facts["time_facts"],  "time")

    non_trivial = [n for n in ans["numbers"] if len(n) > 1]
    q_nums      = [n for n in q_facts["numbers"] if len(n) > 1]
    _check(non_trivial, source["numbers"], q_nums, "num")

    total = len(supported) + len(hallucinated)

    if total == 0:
        score  = 1.0
        reason = "No specific facts in answer to verify."
    else:
        score  = len(supported) / total
        reason = (
            f"{len(supported)} fact(s) supported. "
            f"{len(hallucinated)} fact(s) not in source: {hallucinated[:3]}"
        )

    return {
        "score":              round(score, 4),
        "supported_facts":    supported,
        "hallucinated_facts": hallucinated,
        "reason":             reason,
    }
