import re


def extract_factual_anchors(text: str) -> dict:
    text_lower = text.lower()

    dollar_amounts = re.findall(r'\$[\d,]+(?:\.\d+)?(?:\s*(?:per\s+\w+|annually|monthly))?', text, re.IGNORECASE)
    percentages    = re.findall(r'\d+(?:\.\d+)?\s*%', text)
    time_facts     = re.findall(r'\d+\s*(?:days?|weeks?|months?|years?|hours?)\b', text, re.IGNORECASE)
    numbers        = re.findall(r'\b\d+\b', text)

    return {
        "dollars":      list(set(dollar_amounts)),
        "percentages":  list(set(percentages)),
        "time_facts":   list(set(t.lower().strip() for t in time_facts)),
        "numbers":      list(set(numbers)),
    }


def check_factual_anchors(answer: str, source_context: str) -> dict:
    source  = extract_factual_anchors(source_context)
    ans     = extract_factual_anchors(answer)

    supported    = []
    hallucinated = []

    def _check(ans_list, src_list, label):
        for item in ans_list:
            item_clean = item.lower().strip()
            matched = any(item_clean in s.lower() or s.lower() in item_clean for s in src_list)
            if matched:
                supported.append(f"{label}:{item}")
            else:
                hallucinated.append(f"{label}:{item}")

    _check(ans["dollars"],     source["dollars"],     "dollar")
    _check(ans["percentages"], source["percentages"], "pct")
    _check(ans["time_facts"],  source["time_facts"],  "time")

    # Check standalone numbers only if they are non-trivial (>1 digit avoids false positives)
    non_trivial_nums = [n for n in ans["numbers"] if len(n) > 1]
    _check(non_trivial_nums, source["numbers"], "num")

    total = len(supported) + len(hallucinated)

    if total == 0:
        score = 1.0
        reason = "No specific facts in answer to verify."
    else:
        score = len(supported) / total
        reason = (
            f"{len(supported)} fact(s) matched source context. "
            f"{len(hallucinated)} fact(s) not found in source: {hallucinated[:3]}"
        )

    return {
        "score":             round(score, 4),
        "supported_facts":   supported,
        "hallucinated_facts": hallucinated,
        "reason":            reason,
    }
