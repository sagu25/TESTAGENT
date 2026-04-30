# Explanation Guide — How to Present This Project

Use this guide to explain the project to anyone — from a business stakeholder to a technical team member.

---

## The One-Liner

> **"We built an AI system that automatically tests any RAG application, catches wrong answers using three independent methods — including one that requires zero AI — and shows everything live on a dashboard."**

---

## For a Non-Technical Audience (Business / Management)

### The Problem We Solved

"When you build an AI assistant that answers questions from documents — like an employee policy chatbot — how do you know it's giving correct answers? You cannot manually check every answer every day. And if the AI starts giving wrong answers, your employees get bad information.

We built an automated quality control system. Think of it like a factory quality inspector, but for AI answers. It runs 24/7, catches problems before users do, and shows you a live report."

### What It Does in Plain English

```
Every 2 minutes, our system:

1. Asks the AI assistant a question
2. Reads the answer
3. Checks: Is this answer correct?
           Is it the same as last time?
           Does it match what the document actually says?
4. Gives a score (0 to 1, like a report card)
5. Updates a live dashboard you can see in your browser
```

### Why It Matters

```
Without this system:    A wrong answer could go unnoticed for weeks
With this system:       A wrong answer is caught within 2 minutes
```

---

## For a Technical Audience (Developers / Engineers)

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR                           │
│            APScheduler — fires every 120 seconds            │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │     TEST AGENT      │
        │  1. Probe app       │
        │  2. LLM generates   │
        │     15 test Qs      │
        │  3. Fire 1 Q/cycle  │
        │  4. Save to SQLite  │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  EVALUATOR AGENT    │
        │  Layer 1: Factual   │  ← Pure code, zero LLM
        │  Layer 2: Golden    │  ← Math vs ground truth
        │  Layer 3: LLM Judge │  ← Grounded by golden
        │  Consistency Check  │
        │  Save scores to DB  │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │ STREAMLIT DASHBOARD │
        │  Reads SQLite live  │
        │  Auto-refresh 30s   │
        └─────────────────────┘
```

### The 3-Layer Evaluation (Key Innovation)

**Layer 1 — Factual Anchor Check (Pure Code)**
```python
# Extract all numbers, percentages, dollar amounts from source context
source_facts = extract_factual_anchors(retrieved_context)
# {"numbers": ["15", "5"], "percentages": ["80%"], "dollars": ["$1,500"]}

# Extract same from the app's answer
answer_facts = extract_factual_anchors(app_answer)

# Any fact in the answer NOT in source = potential hallucination
hallucinated = [f for f in answer_facts if f not in source_facts]
score = supported / (supported + hallucinated)
```
Zero LLM calls. Deterministic. Catches wrong numbers instantly.

**Layer 2 — Golden Answer ROUGE-L (Pure Math)**
```python
# Generate reference answer from FULL document (no retrieval gap)
golden = llm(full_document + question)  # runs ONCE, cached forever

# Compute text overlap mathematically
rouge_l = LCS(app_answer, golden_answer) / len(golden_answer)
```
Golden answer is generated with full context — more complete than RAG retrieval.
ROUGE-L is pure math after that. No ongoing LLM dependency.

**Layer 3 — Grounded LLM Judge**
```python
# Judge now has a reference point — much harder to hallucinate
prompt = f"""
  Golden Answer (ground truth): {golden_answer}
  App Answer (evaluate this):   {app_answer}
  Does the app answer contradict the golden answer?
"""
# Judge gives score 0-1 with reasoning
```

**Overall Score Formula**
```
Score = 0.25 × Factual Anchor Score   (Layer 1 — pure code)
      + 0.25 × Golden ROUGE-L         (Layer 2 — pure math)
      + 0.25 × Faithfulness           (Layer 3 — grounded LLM)
      + 0.15 × Relevancy              (Layer 3 — grounded LLM)
      + 0.10 × Completeness           (Layer 3 — grounded LLM)

Hard cap: if Layer 1 score < 0.30 → overall capped at 0.45
(Wrong facts = automatically bad, regardless of other scores)
```

### Consistency Detection

```python
# Every question is asked multiple times across cycles
# All answers stored in SQLite, grouped by question

answers = ["15 days paid leave", "15 annual days", "30 days vacation"]
                                                      ↑ PROBLEM

# Pairwise LLM comparison
for (a, b) in all_pairs(answers):
    result = llm(f"Do these answers contradict? A: {a} B: {b}")
    # Returns: semantic_similarity, contradicts (bool), contradiction_detail

consistency_score = avg_similarity × (1 - contradiction_rate) × (1 - drift)
# Flagged if consistency_score < 0.75
```

---

## For a Demo Audience (Mixed Room)

### Suggested Demo Script (10 minutes)

**Minute 0-2: Setup**
```
"Let me show you a live system. We have three things running:
 - An employee policy chatbot (left screen)
 - An automated tester and evaluator (terminal)
 - A live dashboard (right screen)"
```

**Minute 2-4: Show normal operation**
```
"The system just fired a question automatically — 
 'How many vacation days do employees get?'
 
 The AI answered: '15 days of paid annual leave per year.'
 
 Our evaluator scored it:
   Factual Anchors: 1.00  ← '15' is in the source document
   Golden ROUGE-L:  0.75  ← matches our reference answer
   Faithfulness:    0.95  ← LLM judge confirms it's grounded
   Overall:         0.87  ← GREEN"
```

**Minute 4-7: Inject a wrong answer**
```
"Now watch what happens when I inject a wrong answer.
 I'll change '15 days' to '99 days' in the policy document.
 [make the change]
 
 Wait for next cycle...
 
 The system just detected:
   Factual Anchors: 0.10  ← '99' not in original context
   Golden ROUGE-L:  0.12  ← completely different from reference
   Contradicts Golden: YES
   Overall:         0.22  ← RED — caught immediately"
```

**Minute 7-9: Show consistency detection**
```
"Now look at this question that's been asked 4 times.
 Run 1 said '15 days', Run 2 said '15 days', 
 Run 3 said '99 days', Run 4 said '15 days'.
 
 The system detected the contradiction between Run 1 and Run 3.
 Consistency Score: 0.42 — flagged with a warning.
 The contradiction detail: 'Run 3 states 99 days which contradicts
 Run 1 and Run 4 which state 15 days.'"
```

**Minute 9-10: Close**
```
"Three things make this unique:
 1. It generates its own test questions — no human writes them
 2. It catches wrong answers with code, math, AND AI — not just AI
 3. It runs forever, catching drift and inconsistency over time"
```

---

## Common Questions and Answers

**Q: Why not just use RAGAS or TruLens?**
> "Those tools require you to write test questions and run them manually. Our system reads the app, generates its own questions, and runs continuously on a schedule. The self-configuring + autonomous + multi-layer evaluation combination isn't available out of the box anywhere."

**Q: Why do you use three layers instead of just asking the LLM?**
> "An LLM can hallucinate as a judge too. If the app says '99 days' and the LLM judge has seen similar training data, it might not catch it. Our Layer 1 is pure regex code — it extracts '99' from the answer and '15' from the source and says 'these don't match.' No AI involved, 100% reliable."

**Q: What happens if the RAG app gets better over time?**
> "The scores will go up over time. The consistency score will improve as answers stabilize. The dashboard shows trend lines so you can see the improvement. Golden answers are cached so they act as a stable benchmark."

**Q: Can this work with any RAG app, not just employee policy?**
> "Yes. The test agent probes whatever app you point it at via the RAG_APP_URL in the config. It reads the app's topics and generates questions for that domain automatically. You just change one URL."

**Q: What would it take to make this enterprise-grade?**
> "Three things: replace SQLite with PostgreSQL, add Slack/email alerts when scores drop, and integrate with CI/CD so it triggers on every deploy. The evaluation logic itself is already production-quality."

---

## What the Dashboard Shows

```
Top Row (Layer 1 & 2 — Grounded, Zero LLM):
  Factual Anchor Score  — are the numbers in the answer correct?
  Golden ROUGE-L        — how close is the answer to ground truth?
  Contradicts Golden    — how many runs gave contradictory facts?
  Inconsistent Questions — how many questions have drifting answers?

Middle Row (Layer 3 — LLM Judge, Grounded):
  Overall Score   — weighted average of all 3 layers
  Faithfulness    — is the answer faithful to source context?
  Relevancy       — does the answer address the question?
  Completeness    — is any important information missing?

Charts:
  Line Chart      — all metric scores across every run over time
  Radar Chart     — average breakdown across all 5 dimensions

Per-Question Section (click to expand any question):
  Golden Answer   — the reference answer from full documents
  Run 1, 2, 3... — every answer ever given, with all scores
  Red warnings    — contradictions and hallucinated facts highlighted
```

---

## Score Interpretation

| Score | Color | Meaning | Action |
|---|---|---|---|
| 0.80 – 1.00 | Green | Good — app is working correctly | No action needed |
| 0.60 – 0.79 | Yellow | Warning — some issues detected | Investigate flagged questions |
| 0.00 – 0.59 | Red | Poor — app has serious problems | Fix immediately |

**Consistency Score specifically:**

| Score | Meaning |
|---|---|
| > 0.90 | App gives same answer every time — very stable |
| 0.75 – 0.90 | Minor phrasing variation — acceptable |
| < 0.75 | Flagged — answers are contradicting each other |

---

## Technology Decisions — Why We Chose Each Tool

| Tool | Why This Choice |
|---|---|
| **FastAPI** | Industry standard for Python REST APIs. Swagger UI auto-generated at /docs |
| **TF-IDF (scikit-learn)** | No model download, runs instantly, sufficient for a demo RAG system |
| **Azure OpenAI** | Enterprise-grade, no rate limits, same models as OpenAI but with SLA |
| **APScheduler** | Zero infrastructure — runs inside Python process, no Redis/Celery needed |
| **SQLite** | Zero setup, portable single file, trivial to upgrade to PostgreSQL |
| **Streamlit** | Python-native dashboard in ~200 lines. React would need 10x the code |
| **Plotly** | Interactive charts, works natively with Streamlit |
| **ROUGE-score** | Standard NLP evaluation library — same metric used in academic benchmarks |

---

## Potential Next Steps (If Asked)

| Enhancement | Effort | Value |
|---|---|---|
| Replace SQLite with PostgreSQL | 1 day | Multi-user, concurrent access |
| Add Slack/email alerts | 1 day | Team notified when score drops |
| CI/CD integration | 2 days | Auto-test on every code deploy |
| Replace TF-IDF with ChromaDB | 3 days | Better retrieval, real vector search |
| React dashboard | 1 week | Custom branding, more control |
| Multi-app support | 3 days | Test 5 RAG apps simultaneously |
| Load testing mode | 3 days | Test under concurrent user traffic |
