import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import storage
import metrics
import report_generator


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
    answer = run["answer"]
    run_id = run["id"]

    context_chunks = json.loads(run["retrieved_context"]) if run["retrieved_context"] else []
    context_text = "\n\n".join(
        f"[{c.get('source', 'doc')}] {c.get('text', '')}" for c in context_chunks
    )

    print(f"[EvaluatorAgent] Evaluating run {run_id}: {question[:60]}...")

    faith = metrics.evaluate_faithfulness(question, answer, context_text)
    relev = metrics.evaluate_relevancy(question, answer)
    compl = metrics.evaluate_completeness(question, answer, context_text)

    all_answers = storage.get_all_answers_for_question(question)
    prev_answers = [r["answer"] for r in all_answers if r["answer"] and r["id"] != run_id]
    rouge = 0.0
    if prev_answers:
        rouge_scores = [metrics.rouge_l_score(answer, ref) for ref in prev_answers]
        rouge = sum(rouge_scores) / len(rouge_scores)

    overall = metrics.compute_overall_score(
        faith["score"], relev["score"], compl["score"], rouge
    )

    scores = {
        "faithfulness": faith["score"],
        "faithfulness_reason": faith["reason"],
        "relevancy": relev["score"],
        "relevancy_reason": relev["reason"],
        "completeness": compl["score"],
        "completeness_reason": compl["reason"],
        "rouge_l": rouge,
        "overall": overall,
    }

    storage.save_evaluation(run_id, question, scores)
    print(f"[EvaluatorAgent] Run {run_id} scored: overall={overall:.2f} "
          f"faith={faith['score']:.2f} relev={relev['score']:.2f} "
          f"compl={compl['score']:.2f} rouge={rouge:.2f}")


def _run_consistency_check():
    print("[EvaluatorAgent] Running consistency check across all questions...")

    all_data = storage.get_all_evaluated_data()
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
        print(f"[EvaluatorAgent] Consistency check for: {question[:60]}... ({len(answers)} runs)")
        consistency_data = metrics.compute_consistency_score(answers, question)
        storage.save_consistency_report(question, consistency_data)

        if consistency_data["flagged"]:
            print(f"  ⚠ FLAGGED: consistency={consistency_data['consistency_score']:.2f} "
                  f"contradictions={len(consistency_data['contradiction_details'])}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    storage.init_db()
    run()
