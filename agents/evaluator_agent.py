import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import storage
import metrics
import report_generator
import golden_answer_generator
import multi_judge as mj


def run():
    print("\n[EvaluatorAgent] Starting evaluation...")

    unevaluated = storage.get_unevaluated_runs()
    if not unevaluated:
        print("[EvaluatorAgent] No new runs to evaluate.")
    else:
        for run in unevaluated:
            _evaluate_run(run)

    _run_consistency_check()
    report_generator.generate()
    print("[EvaluatorAgent] Evaluation complete. Report updated.")


def _evaluate_run(run: dict):
    question = run["question"]
    answer   = run["answer"]
    run_id   = run["id"]

    context_chunks = json.loads(run["retrieved_context"]) if run["retrieved_context"] else []
    context_text   = "\n\n".join(
        f"[{c.get('source', 'doc')}] {c.get('text', '')}" for c in context_chunks
    )

    print(f"[EvaluatorAgent] Evaluating run {run_id}: {question[:60]}...")

    # ── LAYER 1: Factual Anchor Check (pure code, no LLM) ────────────────────
    factual = metrics.evaluate_factual_anchors(answer, context_text)
    print(f"  Layer 1 Factual: {factual['score']:.2f} | "
          f"supported={len(factual['supported_facts'])} "
          f"hallucinated={len(factual['hallucinated_facts'])}")
    if factual["hallucinated_facts"]:
        print(f"  [WARN] Hallucinated facts: {factual['hallucinated_facts'][:3]}")

    # ── LAYER 2: Golden Answer (ground truth from full documents) ─────────────
    golden = golden_answer_generator.get_or_generate(question)
    golden_answer = golden["golden_answer"] if golden else ""

    golden_rouge = 0.0
    if golden_answer:
        golden_rouge = metrics.evaluate_golden_rouge_l(answer, golden_answer)
        print(f"  Layer 2 Golden ROUGE-L: {golden_rouge:.2f}")
    else:
        print(f"  Layer 2 Golden: skipped (no source docs accessible)")

    time.sleep(3)  # breathe between LLM calls to stay under rate limit

    # ── LAYER 3: Multi-Judge Consensus ────────────────────────────────────────
    available_judges = len(mj._get_all_clients())
    judge_label = f"Multi-Judge ({available_judges} models)" if available_judges > 1 else "Single Judge"

    if golden_answer:
        faith = mj.multi_judge_faithfulness(question, answer, context_text, golden_answer)
        relev = mj.multi_judge_relevancy(question, answer, golden_answer)
        compl = mj.multi_judge_completeness(question, answer, golden_answer)
    else:
        from metrics import evaluate_faithfulness, evaluate_relevancy, evaluate_completeness
        faith = evaluate_faithfulness(question, answer, context_text)
        faith["contradicts_golden"]   = False
        faith["contradiction_detail"] = ""
        faith["disputed"]             = False
        relev = evaluate_relevancy(question, answer)
        compl = evaluate_completeness(question, answer, context_text)

    print(f"  Layer 3 [{judge_label}]: faith={faith['score']:.2f} "
          f"relev={relev['score']:.2f} compl={compl['score']:.2f}")

    if faith.get("disputed"):
        print(f"  [DISPUTED] Judges disagree on faithfulness by {faith.get('disagreement', 0):.2f}")
    if faith.get("contradicts_golden"):
        print(f"  [CONTRADICTION] CONTRADICTS GOLDEN ANSWER: {faith.get('contradiction_detail', '')}")

    # ── Cross-run ROUGE-L (consistency signal across previous runs) ───────────
    all_prev = storage.get_all_answers_for_question(question)
    prev_answers = [r["answer"] for r in all_prev if r["answer"] and r["id"] != run_id]
    cross_rouge = 0.0
    if prev_answers:
        scores = [metrics.rouge_l_score(answer, ref) for ref in prev_answers]
        cross_rouge = sum(scores) / len(scores)

    # ── Final Score (4-layer formula) ─────────────────────────────────────────
    overall = metrics.compute_overall_score(
        faithfulness        = faith["score"],
        relevancy           = relev["score"],
        completeness        = compl["score"],
        rouge_l             = cross_rouge,
        factual_anchor_score= factual["score"],
        golden_rouge_l      = golden_rouge if golden_answer else None,
    )

    scores = {
        "faithfulness":          faith["score"],
        "faithfulness_reason":   faith["reason"],
        "relevancy":             relev["score"],
        "relevancy_reason":      relev["reason"],
        "completeness":          compl["score"],
        "completeness_reason":   compl["reason"],
        "rouge_l":               cross_rouge,
        "factual_anchor_score":  factual["score"],
        "factual_supported":     factual["supported_facts"],
        "factual_hallucinated":  factual["hallucinated_facts"],
        "golden_rouge_l":        golden_rouge,
        "contradicts_golden":    faith.get("contradicts_golden", False),
        "contradiction_detail":  faith.get("contradiction_detail", ""),
        "overall":               overall,
    }

    storage.save_evaluation(run_id, question, scores)
    print(f"  [OK] Overall Score: {overall:.2f}")


def _run_consistency_check():
    print("[EvaluatorAgent] Running consistency check across all questions...")

    all_data    = storage.get_all_evaluated_data()
    evaluations = all_data["evaluations"]

    questions_seen = {}
    for ev in evaluations:
        q = ev["question"]
        if q not in questions_seen:
            questions_seen[q] = []
        questions_seen[q].append(ev["answer"])

    for question, answers in questions_seen.items():
        if len(answers) < 2:
            continue
        print(f"  Checking: {question[:60]}... ({len(answers)} runs)")
        consistency_data = metrics.compute_consistency_score(answers, question)
        storage.save_consistency_report(question, consistency_data)

        if consistency_data["flagged"]:
            print(f"  [FLAGGED] consistency={consistency_data['consistency_score']:.2f} "
                  f"contradictions={len(consistency_data['contradiction_details'])}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    storage.init_db()
    run()
