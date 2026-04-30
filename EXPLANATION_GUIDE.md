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

## Mathematical Formulas — What We Use and Why

This section explains every formula in the system, what it measures, and why it was chosen over alternatives.

---

### Formula 1 — Factual Anchor Score (Layer 1)

**What it is:** Pure code. No AI. Extracts specific facts from text using pattern matching and compares them.

**The Math:**

```
Step 1: Extract facts from source context
  source_facts = regex_extract(retrieved_context)
  → numbers:     ["15", "5", "3"]
  → percentages: ["80%", "20%", "5%"]
  → dollar amts: ["$1,500", "$100"]
  → time facts:  ["15 days", "6 months", "3 weeks"]

Step 2: Extract facts from app's answer
  answer_facts = regex_extract(app_answer)
  → numbers:     ["99", "5"]       ← "99" is suspicious
  → time facts:  ["99 days"]       ← not in source!

Step 3: Compare
  supported    = facts in answer that ALSO exist in source
  hallucinated = facts in answer that DO NOT exist in source

Step 4: Score
  Factual Anchor Score = supported / (supported + hallucinated)
                       = 1 / (1 + 1)
                       = 0.50  ← answer has 1 correct + 1 hallucinated fact
```

**Real example:**
```
Source says:    "Employees get 15 days of paid leave"
App answers:    "Employees get 99 days of paid leave"

supported    = []        (none match)
hallucinated = ["99"]    (not in source)
Score        = 0 / 1 = 0.00  ← CAUGHT IMMEDIATELY
```

**Why this formula:**
- It is the only metric in the system that is 100% deterministic
- An LLM cannot argue with it — "99 is not in the source document" is a fact
- Catches numerical hallucinations that LLM judges sometimes miss
- Runs in milliseconds — no API call, no cost

---

### Formula 2 — ROUGE-L Score (Layer 2, against Golden Answer)

**What it is:** Recall-Oriented Understudy for Gisting Evaluation. Measures how much of the reference answer appears in the evaluated answer, using the longest common subsequence.

**The Math:**

```
ROUGE-L = F1 score of the Longest Common Subsequence (LCS)

  Precision = LCS(answer, golden) / length(answer)
  Recall    = LCS(answer, golden) / length(golden)
  ROUGE-L   = 2 × Precision × Recall / (Precision + Recall)
```

**Simple example:**
```
Golden Answer: "Employees receive 15 days of paid annual leave per year"
App Answer:    "Employees get 15 days paid leave"

LCS = "Employees ... 15 days ... paid ... leave"   (length = 5 key tokens)

Precision = 5 / 6  = 0.83   (5 of 6 answer tokens in LCS)
Recall    = 5 / 9  = 0.56   (5 of 9 golden tokens in LCS)
ROUGE-L   = 2×0.83×0.56 / (0.83+0.56) = 0.67
```

**Wrong answer example:**
```
Golden Answer: "Employees receive 15 days of paid annual leave per year"
App Answer:    "Employees get 99 days of vacation"

LCS = "Employees ... days"   (length = 2 tokens only)

Precision = 2 / 5  = 0.40
Recall    = 2 / 9  = 0.22
ROUGE-L   = 2×0.40×0.22 / (0.40+0.22) = 0.28  ← LOW — wrong answer detected
```

**Why ROUGE-L specifically (not ROUGE-1 or ROUGE-2):**
```
ROUGE-1  counts individual word overlaps      → misses word order
ROUGE-2  counts 2-word phrase overlaps        → misses paraphrasing
ROUGE-L  uses longest common subsequence      → captures meaning flow
         handles paraphrasing + word order

Example: "15 days paid" vs "paid for 15 days"
  ROUGE-1: high (same words)
  ROUGE-2: low (different pairs)
  ROUGE-L: medium (correct — similar but reordered)
```

**Why against Golden Answer (not previous runs):**
```
Previous approach:  ROUGE-L vs other app answers
  Problem: if app always says "99 days", ROUGE-L is 1.0 (consistent wrong answer)

New approach:       ROUGE-L vs Golden Answer (full document reference)
  Benefit: even if app is consistently wrong, score is low vs the correct reference
```

---

### Formula 3 — Overall Score (Weighted Combination)

**The Math:**

```
Overall Score =
  0.25 × Factual Anchor Score    (Layer 1 — pure code)
+ 0.25 × Golden ROUGE-L          (Layer 2 — pure math)
+ 0.25 × Faithfulness Score      (Layer 3 — grounded LLM)
+ 0.15 × Relevancy Score         (Layer 3 — grounded LLM)
+ 0.10 × Completeness Score      (Layer 3 — grounded LLM)

HARD RULE: If Factual Anchor Score < 0.30
           → cap Overall Score at 0.45 regardless of other scores
```

**Why these weights:**
```
Factual Anchors (0.25):  Highest priority per unit — deterministic, no hallucination risk
Golden ROUGE-L  (0.25):  Mathematical ground truth — equally important
Faithfulness    (0.25):  Core RAG metric — is answer grounded in source?
Relevancy       (0.15):  Important but secondary — answer can be relevant but wrong
Completeness    (0.10):  Least weight — missing info is bad but less bad than wrong info

Total = 1.00
```

**Why the hard cap:**
```
Scenario without cap:
  Factual Anchor: 0.10  (app said "99 days" — wrong!)
  Golden ROUGE-L: 0.15  (very different from reference)
  Faithfulness:   0.90  (LLM judge was fooled)
  Relevancy:      0.95  (answer IS about leave days)
  Completeness:   0.80
  
  Without cap: 0.25×0.10 + 0.25×0.15 + 0.25×0.90 + 0.15×0.95 + 0.10×0.80
             = 0.025 + 0.0375 + 0.225 + 0.1425 + 0.08
             = 0.51  ← would show as YELLOW (warning), not RED!

  With cap:   Factual < 0.30 → cap at 0.45  ← RED — correctly flagged
```

The hard cap ensures wrong factual answers are always flagged red, even if the LLM judge gives high scores.

---

### Formula 4 — Consistency Score (Cross-Run)

**What it is:** Measures how stable the app's answers are across multiple runs of the same question.

**The Math:**

```
Given N answers to the same question:
  answers = [a1, a2, a3, ..., aN]

Step 1: Pairwise Semantic Similarity
  For every pair (ai, aj):
    sim(i,j) = LLM judge score for semantic similarity  [0.0 to 1.0]
  
  avg_similarity = mean of all pairwise sim scores

Step 2: Contradiction Rate
  For every pair (ai, aj):
    contradicts(i,j) = LLM judge says YES/NO
  
  contradiction_rate = contradicting_pairs / total_pairs

Step 3: Drift Score (first answer vs latest answer)
  drift = 1 - sim(a1, aN)
  → 0.0 = no drift (same answer), 1.0 = completely different

Step 4: Consistency Score
  consistency = avg_similarity
              × (1 - contradiction_rate)
              × (1 - 0.5 × drift)

  Flag as INCONSISTENT if consistency < 0.75
```

**Real example:**
```
Question: "How many vacation days do employees get?"

Run 1: "15 days of paid leave"
Run 2: "15 annual leave days"
Run 3: "30 days vacation"         ← WRONG
Run 4: "15 days paid annually"

Pairwise similarities:
  (1,2): 0.92  no contradiction
  (1,3): 0.12  CONTRADICTS  "15 days" vs "30 days"
  (1,4): 0.91  no contradiction
  (2,3): 0.11  CONTRADICTS
  (2,4): 0.89  no contradiction
  (3,4): 0.13  CONTRADICTS

avg_similarity     = (0.92+0.12+0.91+0.11+0.89+0.13) / 6 = 0.51
contradiction_rate = 3 contradicting / 6 total = 0.50
drift              = 1 - sim(Run1, Run4) = 1 - 0.91 = 0.09

consistency = 0.51 × (1 - 0.50) × (1 - 0.5×0.09)
            = 0.51 × 0.50 × 0.955
            = 0.24  ← FAR below 0.75 → FLAGGED
```

**Why multiply instead of average:**
```
Multiplying the three factors means ALL three must be good for a high score.
If any one factor is bad, the whole score drops.

Averaging would allow: high similarity + high contradiction = medium score
  (which makes no sense — you can't be similar AND contradictory)

Multiplying enforces: if contradiction_rate is high → score collapses
```

---

### Formula 5 — Faithfulness Score (Layer 3, LLM Judge)

**What it is:** The LLM evaluates whether every claim in the answer is supported by the retrieved context AND matches the golden answer.

**The Prompt Structure:**
```
Inputs to LLM:
  1. The question asked
  2. Retrieved context (what RAG app used)
  3. Golden answer (full-document reference)
  4. App's answer (what we're evaluating)

LLM is asked:
  "Does the app answer contradict the golden answer?"
  "Are there facts in the answer not in the retrieved context?"
  "Score: 0.0 (contradicts golden) to 1.0 (fully faithful)"
```

**Why LLM judge is still needed (despite Layers 1 and 2):**
```
Layer 1 catches:  Wrong numbers ("99 days" vs "15 days")
Layer 2 catches:  Low text overlap with ground truth

But neither catches:
  "The policy allows unlimited leave"    ← no specific number, but completely wrong
  "Employees should consult their manager about leave"  ← vague, misleading
  "The company offers leave benefits"    ← too generic, avoids the question

LLM judge catches these semantic errors that code and math miss.
```

**Grounding makes it reliable:**
```
Without golden answer (old approach):
  LLM sees: question + context + answer
  LLM might: be fooled by confident-sounding wrong answers

With golden answer (new approach):
  LLM sees: question + context + VERIFIED REFERENCE + answer
  LLM judges: "Does this match the verified reference?"
  Much harder to fool — reference is the anchor
```

---

### How All 5 Formulas Work Together

```
One evaluation run produces 5 independent signals:

Signal              Source              Can be fooled?
─────────────────   ─────────────────   ──────────────
Factual Anchor      Pure regex code     NO — deterministic
Golden ROUGE-L      LCS algorithm       NO — deterministic
Faithfulness        LLM + golden ref    HARD — has reference anchor
Relevancy           LLM + golden ref    HARD — has reference anchor
Completeness        LLM + golden ref    HARD — has reference anchor

A wrong answer must fool ALL 5 signals to get a high overall score.
That is extremely unlikely when 2 of the 5 are pure math.
```

**Detection coverage:**

| Problem Type | Caught By |
|---|---|
| Wrong number (15 → 99) | Layer 1 (Factual Anchor) immediately |
| Off-topic answer | Layer 2 (low ROUGE-L) + Layer 3 (Relevancy) |
| Vague non-answer | Layer 2 (low ROUGE-L) + Layer 3 (Completeness) |
| Hallucinated policy | Layer 3 (Faithfulness) against golden |
| Inconsistent across runs | Consistency Score (pairwise comparison) |
| Answer drift over time | Drift Score (first vs latest run) |
| Correct phrasing, wrong facts | Layer 1 + Hard cap on overall score |

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
