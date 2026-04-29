import os
import json
from datetime import datetime
import storage

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")


def _score_color(score: float) -> str:
    if score is None:
        return "#999"
    if score >= 0.80:
        return "#22c55e"
    if score >= 0.60:
        return "#f59e0b"
    return "#ef4444"


def _score_label(score: float) -> str:
    if score is None:
        return "N/A"
    if score >= 0.80:
        return "GOOD"
    if score >= 0.60:
        return "WARN"
    return "POOR"


def generate():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    data = storage.get_all_evaluated_data()
    evaluations = data["evaluations"]
    consistency_reports = data["consistency"]

    if not evaluations:
        print("[ReportGenerator] No data to report yet.")
        return

    avg = lambda key: round(
        sum(e[key] for e in evaluations if e[key] is not None) /
        max(1, sum(1 for e in evaluations if e[key] is not None)), 3
    )

    overall_avg = avg("overall_score")
    faith_avg = avg("faithfulness")
    relev_avg = avg("relevancy")
    compl_avg = avg("completeness")
    rouge_avg = avg("rouge_l")

    flagged_questions = [c for c in consistency_reports if c["flagged"]]
    total_runs = len(evaluations)
    unique_questions = len(set(e["question"] for e in evaluations))

    questions_map = {}
    for ev in evaluations:
        q = ev["question"]
        if q not in questions_map:
            questions_map[q] = []
        questions_map[q].append(ev)

    consistency_map = {c["question"]: c for c in consistency_reports}

    question_rows = ""
    for q, runs in questions_map.items():
        cons = consistency_map.get(q, {})
        cons_score = cons.get("consistency_score")
        cons_color = _score_color(cons_score)
        cons_label = _score_label(cons_score) if cons_score else "1 run"
        flagged = cons.get("flagged", False)
        flag_badge = '<span style="background:#ef4444;color:white;padding:2px 8px;border-radius:12px;font-size:11px;margin-left:8px;">⚠ INCONSISTENT</span>' if flagged else ""

        run_rows = ""
        for i, run in enumerate(runs):
            o = run.get("overall_score")
            f = run.get("faithfulness")
            r = run.get("relevancy")
            c = run.get("completeness")
            rl = run.get("rouge_l")
            run_time = run.get("run_time", run.get("timestamp", ""))[:19]
            answer_preview = (run.get("answer") or "")[:200]
            if len(run.get("answer") or "") > 200:
                answer_preview += "..."

            fs = f"{f:.2f}" if f is not None else "N/A"
            rs = f"{r:.2f}" if r is not None else "N/A"
            cs = f"{c:.2f}" if c is not None else "N/A"
            rls = f"{rl:.2f}" if rl is not None else "N/A"
            os_ = f"{o:.2f}" if o is not None else "N/A"

            run_rows += f"""
            <tr>
              <td style="padding:8px;border:1px solid #e2e8f0;font-weight:600;">Run {i+1}<br><small style="color:#64748b;font-weight:400;">{run_time}</small></td>
              <td style="padding:8px;border:1px solid #e2e8f0;font-size:13px;">{answer_preview}</td>
              <td style="padding:8px;border:1px solid #e2e8f0;text-align:center;color:{_score_color(f)};font-weight:700;">{fs}</td>
              <td style="padding:8px;border:1px solid #e2e8f0;text-align:center;color:{_score_color(r)};font-weight:700;">{rs}</td>
              <td style="padding:8px;border:1px solid #e2e8f0;text-align:center;color:{_score_color(c)};font-weight:700;">{cs}</td>
              <td style="padding:8px;border:1px solid #e2e8f0;text-align:center;color:{_score_color(rl)};font-weight:700;">{rls}</td>
              <td style="padding:8px;border:1px solid #e2e8f0;text-align:center;background:{_score_color(o)};color:white;font-weight:700;">{os_}</td>
            </tr>"""

        contradiction_html = ""
        if cons.get("contradiction_details"):
            details = cons["contradiction_details"]
            if isinstance(details, str):
                details = json.loads(details)
            for d in details:
                contradiction_html += f"""
                <div style="margin-top:6px;padding:8px;background:#fef2f2;border-left:3px solid #ef4444;border-radius:4px;font-size:13px;">
                  <strong>Run {d.get('run_a')} vs Run {d.get('run_b')}:</strong> {d.get('detail', '')}
                </div>"""

        question_rows += f"""
        <div style="margin-bottom:32px;border:1px solid #e2e8f0;border-radius:12px;overflow:hidden;">
          <div style="padding:16px 20px;background:#f8fafc;border-bottom:1px solid #e2e8f0;">
            <div style="font-size:15px;font-weight:600;color:#1e293b;">{q}{flag_badge}</div>
            <div style="margin-top:8px;display:flex;gap:16px;font-size:13px;">
              <span>Runs: <strong>{len(runs)}</strong></span>
              <span>Consistency: <strong style="color:{cons_color};">{f'{cons_score:.2f}' if cons_score is not None else '—'}</strong> {f'<span style="color:{cons_color};">({cons_label})</span>' if cons_score is not None else ''}</span>
              <span>Drift: <strong>{f'{cons.get("drift_score", 0):.2f}' if cons.get("drift_score") is not None else '—'}</strong></span>
              <span>Contradictions: <strong style="color:{'#ef4444' if cons.get('contradiction_rate', 0) > 0 else '#22c55e'};">{f'{cons.get("contradiction_rate", 0)*100:.0f}%'}</strong></span>
            </div>
          </div>
          <div style="overflow-x:auto;">
            <table style="width:100%;border-collapse:collapse;font-size:13px;">
              <thead>
                <tr style="background:#f1f5f9;">
                  <th style="padding:10px;border:1px solid #e2e8f0;text-align:left;">Run</th>
                  <th style="padding:10px;border:1px solid #e2e8f0;text-align:left;">Answer</th>
                  <th style="padding:10px;border:1px solid #e2e8f0;text-align:center;">Faith.</th>
                  <th style="padding:10px;border:1px solid #e2e8f0;text-align:center;">Relev.</th>
                  <th style="padding:10px;border:1px solid #e2e8f0;text-align:center;">Compl.</th>
                  <th style="padding:10px;border:1px solid #e2e8f0;text-align:center;">ROUGE-L</th>
                  <th style="padding:10px;border:1px solid #e2e8f0;text-align:center;">Overall</th>
                </tr>
              </thead>
              <tbody>{run_rows}</tbody>
            </table>
          </div>
          {f'<div style="padding:12px 20px;background:#fff7f7;">{contradiction_html}</div>' if contradiction_html else ''}
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RAG Evaluation Report</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f8fafc; color: #1e293b; }}
    .container {{ max-width: 1400px; margin: 0 auto; padding: 32px 24px; }}
    .header {{ margin-bottom: 32px; }}
    .header h1 {{ font-size: 28px; font-weight: 700; color: #0f172a; }}
    .header p {{ color: #64748b; margin-top: 4px; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; margin-bottom: 32px; }}
    .stat-card {{ background: white; border-radius: 12px; padding: 20px; border: 1px solid #e2e8f0; text-align: center; }}
    .stat-value {{ font-size: 32px; font-weight: 700; }}
    .stat-label {{ font-size: 13px; color: #64748b; margin-top: 4px; }}
    .section-title {{ font-size: 20px; font-weight: 600; margin-bottom: 16px; color: #0f172a; }}
    .alert-box {{ background: #fef2f2; border: 1px solid #fecaca; border-radius: 12px; padding: 16px 20px; margin-bottom: 32px; }}
    .alert-box h3 {{ color: #dc2626; margin-bottom: 8px; }}
    .alert-item {{ font-size: 13px; padding: 4px 0; color: #7f1d1d; }}
  </style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>RAG System Evaluation Report</h1>
    <p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC &nbsp;|&nbsp; Total Runs: {total_runs} &nbsp;|&nbsp; Unique Questions: {unique_questions}</p>
  </div>

  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-value" style="color:{_score_color(overall_avg)};">{overall_avg:.2f}</div>
      <div class="stat-label">Overall Score</div>
    </div>
    <div class="stat-card">
      <div class="stat-value" style="color:{_score_color(faith_avg)};">{faith_avg:.2f}</div>
      <div class="stat-label">Faithfulness</div>
    </div>
    <div class="stat-card">
      <div class="stat-value" style="color:{_score_color(relev_avg)};">{relev_avg:.2f}</div>
      <div class="stat-label">Relevancy</div>
    </div>
    <div class="stat-card">
      <div class="stat-value" style="color:{_score_color(compl_avg)};">{compl_avg:.2f}</div>
      <div class="stat-label">Completeness</div>
    </div>
    <div class="stat-card">
      <div class="stat-value" style="color:{_score_color(rouge_avg)};">{rouge_avg:.2f}</div>
      <div class="stat-label">ROUGE-L</div>
    </div>
    <div class="stat-card">
      <div class="stat-value" style="color:{'#ef4444' if flagged_questions else '#22c55e'};">{len(flagged_questions)}</div>
      <div class="stat-label">Inconsistent Questions</div>
    </div>
  </div>

  {'<div class="alert-box"><h3>⚠ Consistency Alerts</h3>' + ''.join(f'<div class="alert-item">• {c["question"][:80]}... — consistency: {c["consistency_score"]:.2f}, contradictions: {c["contradiction_rate"]*100:.0f}%</div>' for c in flagged_questions) + '</div>' if flagged_questions else ''}

  <div class="section-title">Question-Level Comparison (All Runs)</div>
  {question_rows}
</div>
</body>
</html>"""

    report_path = os.path.join(REPORTS_DIR, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    json_path = os.path.join(REPORTS_DIR, "report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat(),
            "summary": {
                "total_runs": total_runs,
                "unique_questions": unique_questions,
                "overall_avg": overall_avg,
                "faithfulness_avg": faith_avg,
                "relevancy_avg": relev_avg,
                "completeness_avg": compl_avg,
                "rouge_l_avg": rouge_avg,
                "flagged_questions": len(flagged_questions),
            },
            "evaluations": evaluations,
            "consistency": consistency_reports,
        }, f, indent=2)

    print(f"[ReportGenerator] Report saved to {report_path}")
