import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import storage
import metrics
import report_generator
import golden_answer_generator
import multi_judge as mj
import cache
import eval_version
import retrieval_metrics


EVAL_VERSION = eval_version.get_version()


def run():
    print(f"\n[EvaluatorAgent] Starting evaluation (eval_version={EVAL_VERSION})...")
    cache.init_cache_table()

    unevaluated = storage.get_unevaluated_runs()
    if not unevaluated:
        print("[EvaluatorAgent] No new runs to evaluate.")
    else:
        for run in unevaluated:
            _evaluate_run(run)

    _run_consistency_check()
    report_generator.generate()
    print("[EvaluatorAgent] Evaluation complete. Report updated.")

    stats = cache.cache_stats()
    print(f"[EvaluatorAgent] Cache stats: {stats['total']} entries — {stats['by_metric']}")


def _evaluate_run(run: dict):
    question = run["question"]
    answer   = run["answer"]
    run_id   = run["id"]

    context_chunks = json.loads(run["retrieved_context"]) if run["retrieved_context"] else []
    context_text   = "\n\n".join(
        f"[{c.get('source','doc')}] {c.get('text','')}" for c in context_chunks
    )

    print(f"\n[EvaluatorAgent] Evaluating run {run_id}: {question[:60]}...")

    # ── LAYER 0: Retrieval Quality (context precision + recall) ───────────────
    ctx_precision = {"score": None, "reason": "skipped"}
    ctx_recall    = {"score": None, "reason": "skipped"}

    if context_chunks:
        print(f"  Layer 0 Retrieval: scoring {len(context_chunks)} chunks...")
        ctx_precision = retrieval_metrics.evaluate_context_precision(question, context_chunks)
        print(f"  Context Precision: {ctx_precision['score']:.2f} "
              f"({ctx_precision['relevant_count']}/{ctx_precision['total_chunks']} chunks relevant)")

    # ── LAYER 1: Factual Anchor Check (pure code) ─────────────────────────────
    factual = metrics.evaluate_factual_anchors(answer, context_text, question)
    print(f"  Layer 1 Factual: {factual['score']:.2f} | "
          f"supported={len(factual['supported_facts'])} "
          f"hallucinated={len(factual['hallucinated_facts'])}")
    if factual["hallucinated_facts"]:
        print(f"  [WARN] Hallucinated facts: {factual['hallucinated_facts'][:3]}")

    # ── LAYER 2: Golden Answer + ROUGE-L ─────────────────────────────────────
    golden        = golden_answer_generator.get_or_generate(question)
    golden_answer = golden["golden_answer"] if golden else ""
    golden_rouge  = 0.0

    if golden_answer:
        golden_rouge = metrics.evaluate_golden_rouge_l(answer, golden_answer)
        print(f"  Layer 2 Golden ROUGE-L: {golden_rouge:.2f}")

        # Context Recall (needs golden answer)
        if context_chunks:
            ctx_recall = retrieval_metrics.evaluate_context_recall(
                question, golden_answer, context_chunks
            )
            print(f"  Context Recall: {ctx_recall['score']:.2f} — {ctx_recall['reason'][:60]}")

    # ── LAYER 3: Multi-Judge (parallel via ThreadPoolExecutor) ────────────────
    cache_key      = cache.make_key(question, answer, golden_answer, EVAL_VERSION)
    judge_count    = len(mj._get_all_clients())
    judge_label    = f"Multi-Judge ({judge_count} models)" if judge_count > 1 else "Single Judge"

    time.sleep(2)  # small buffer to avoid rate limits between layers

    if golden_answer:
        def run_faith():
            cached = cache.get(cache_key, "faithfulness")
            if cached:
                print(f"  [CACHE HIT] faithfulness")
                return cached
            result = mj.multi_judge_faithfulness(question, answer, context_text, golden_answer)
            cache.set(cache_key, "faithfulness", result)
            return result

        def run_relev():
            cached = cache.get(cache_key, "relevancy")
            if cached:
                print(f"  [CACHE HIT] relevancy")
                return cached
            result = mj.multi_judge_relevancy(question, answer, golden_answer)
            cache.set(cache_key, "relevancy", result)
            return result

        def run_compl():
            cached = cache.get(cache_key, "completeness")
            if cached:
                print(f"  [CACHE HIT] completeness")
                return cached
            result = mj.multi_judge_completeness(question, answer, golden_answer)
            cache.set(cache_key, "completeness", result)
            return result

        # Run all 3 judges in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_faith = executor.submit(run_faith)
            future_relev = executor.submit(run_relev)
            future_compl = executor.submit(run_compl)
            faith = future_faith.result()
            relev = future_relev.result()
            compl = future_compl.result()
    else:
        faith = metrics.evaluate_faithfulness(question, answer, context_text)
        faith["contradicts_golden"] = False
        faith["contradiction_detail"] = ""
        faith["disputed"] = False
        relev = metrics.evaluate_relevancy(question, answer)
        compl = metrics.evaluate_completeness(question, answer, context_text)

    print(f"  Layer 3 [{judge_label}]: "
          f"faith={faith['score']:.2f} relev={relev['score']:.2f} compl={compl['score']:.2f}")

    if faith.get("disputed"):
        print(f"  [DISPUTED] Judges disagree by {faith.get('disagreement', 0):.2f}")
    if faith.get("contradicts_golden"):
        print(f"  [CONTRADICTION] {faith.get('contradiction_detail', '')}")

    # ── Cross-run ROUGE-L (consistency signal) ─────────────────────────────────
    all_prev   = storage.get_all_answers_for_question(question)
    prev_ans   = [r["answer"] for r in all_prev if r["answer"] and r["id"] != run_id]
    cross_rouge = 0.0
    if prev_ans:
        scores      = [metrics.rouge_l_score(answer, ref) for ref in prev_ans]
        cross_rouge = sum(scores) / len(scores)

    # ── Final Score ────────────────────────────────────────────────────────────
    overall = metrics.compute_overall_score(
        faithfulness         = faith["score"],
        relevancy            = relev["score"],
        completeness         = compl["score"],
        rouge_l              = cross_rouge,
        factual_anchor_score = factual["score"],
        golden_rouge_l       = golden_rouge if golden_answer else None,
    )

    scores = {
        "faithfulness":            faith["score"],
        "faithfulness_reason":     faith.get("reason", ""),
        "relevancy":               relev["score"],
        "relevancy_reason":        relev.get("reason", ""),
        "completeness":            compl["score"],
        "completeness_reason":     compl.get("reason", ""),
        "rouge_l":                 cross_rouge,
        "factual_anchor_score":    factual["score"],
        "factual_supported":       factual["supported_facts"],
        "factual_hallucinated":    factual["hallucinated_facts"],
        "golden_rouge_l":          golden_rouge,
        "contradicts_golden":      faith.get("contradicts_golden", False),
        "contradiction_detail":    faith.get("contradiction_detail", ""),
        "eval_version":            EVAL_VERSION,
        "judge_count":             judge_count,
        "judge_disputed":          faith.get("disputed", False),
        "context_precision":       ctx_precision.get("score"),
        "context_recall":          ctx_recall.get("score"),
        "context_precision_reason":ctx_precision.get("reason", ""),
        "context_recall_reason":   ctx_recall.get("reason", ""),
        "overall":                 overall,
    }

    storage.save_evaluation(run_id, question, scores)
    print(f"  [OK] Overall: {overall:.2f} | "
          f"CtxPrec: {ctx_precision.get('score', 'N/A')} | "
          f"CtxRecall: {ctx_recall.get('score', 'N/A')}")


def _run_consistency_check():
    print("\n[EvaluatorAgent] Running consistency check...")
    all_data    = storage.get_all_evaluated_data()
    evaluations = all_data["evaluations"]

    questions_seen: dict[str, list[str]] = {}
    for ev in evaluations:
        q = ev["question"]
        if q not in questions_seen:
            questions_seen[q] = []
        questions_seen[q].append(ev["answer"])

    for question, answers in questions_seen.items():
        if len(answers) < 2:
            continue
        print(f"  Checking: {question[:60]}... ({len(answers)} runs)")
        data = metrics.compute_consistency_score(answers, question)
        storage.save_consistency_report(question, data)
        if data["flagged"]:
            print(f"  [FLAGGED] consistency={data['consistency_score']:.2f} "
                  f"contradictions={len(data['contradiction_details'])}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    storage.init_db()
    run()
