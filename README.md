# RAG Test & Evaluator Agent System

An autonomous multi-agent system that continuously tests any RAG (Retrieval-Augmented Generation) application, detects wrong answers, catches inconsistencies across runs, and displays everything live on a Streamlit dashboard — **without any hardcoded expected answers.**

---

## What Problem Does This Solve?

When you build a RAG application, you need to answer three critical questions:

1. **Is the app giving correct answers?** (Faithfulness)
2. **Is the app giving the same answer every time?** (Consistency)
3. **Is the app missing important information?** (Completeness)

Traditionally, you would hire people to manually test this — writing expected answers, comparing outputs, and tracking changes over time. That is slow, expensive, and does not scale.

**This system does all of that automatically, around the clock, with zero human effort after setup.**

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR                                │
│              Triggers the pipeline every 3 minutes                  │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │       TEST AGENT        │
          │                         │
          │  Step 1 (Once):         │
          │  Analyze app → Generate │
          │  15 test questions      │
          │                         │
          │  Step 2 (Every cycle):  │
          │  Fire 1 question to app │
          │  Save Q + Answer to DB  │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │    EVALUATOR AGENT      │
          │                         │
          │  Score each answer:     │
          │  • Faithfulness         │
          │  • Relevancy            │
          │  • Completeness         │
          │  • ROUGE-L              │
          │                         │
          │  Consistency Check:     │
          │  Compare all runs of    │
          │  same question          │
          │  Flag contradictions    │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │    STREAMLIT DASHBOARD  │
          │                         │
          │  Live scores & charts   │
          │  Consistency alerts     │
          │  Per-question comparison│
          │  Auto-refreshes 30s     │
          └─────────────────────────┘
```

---

## Project Structure

```
eval_system/
│
├── rag_app/                    ← The mock RAG application (being tested)
│   ├── main.py                 ← FastAPI server with /query endpoint
│   ├── documents.py            ← Employee policy documents (6 policies)
│   └── retriever.py            ← TF-IDF document search engine
│
├── agents/                     ← The two AI agents
│   ├── test_agent.py           ← Generates questions and fires them
│   └── evaluator_agent.py      ← Scores answers and checks consistency
│
├── dashboard.py                ← Streamlit live dashboard
├── llm_client.py               ← Unified LLM client (Groq / Azure / Grok)
├── metrics.py                  ← All scoring formulas and LLM judge calls
├── storage.py                  ← SQLite database operations
├── orchestrator.py             ← Scheduler that runs everything every 3 min
├── report_generator.py         ← Generates static HTML report
├── start_rag_app.py            ← Starts the RAG server
├── requirements.txt            ← Python dependencies
└── .env.example                ← Environment variable template
```

---

## The RAG Application (What Gets Tested)

The mock RAG app simulates a real-world **Employee Policy Document Assistant**. It covers:

| Policy Document | Topics Covered |
|---|---|
| Annual Leave Policy | Leave days, accrual, carry-forward rules |
| Remote Work Policy | Eligibility, remote days allowed, equipment |
| Code of Conduct | Professional behavior, conflicts of interest |
| Employee Benefits | Health insurance, 401k, wellness allowance |
| Performance Review | Rating scale, salary increases, PIPs |
| Expense Reimbursement | Meal limits, travel rules, submission process |

### How the RAG App Works Internally

```
User Question
     │
     ▼
TF-IDF Search Engine
(scikit-learn)
     │
     Finds top 4 most relevant
     chunks from policy documents
     │
     ▼
LLM (Groq / Azure)
     │
     Reads the retrieved chunks
     Generates an answer based
     ONLY on those chunks
     │
     ▼
Returns: Answer + Retrieved Context + Sources
```

**Why TF-IDF?**
TF-IDF (Term Frequency-Inverse Document Frequency) is a mathematical formula that scores how relevant each document chunk is to the question. It requires no model download, no GPU, and runs instantly. It is the right choice for a demo system.

---

## Agent 1 — Test Agent

**File:** `agents/test_agent.py`

**Purpose:** Automatically discover what the app does and test it continuously.

### How It Works

#### Step 1 — App Analysis (Runs Only Once)

```python
# Test Agent probes the app
GET /topics → ["Annual Leave Policy", "Remote Work Policy", ...]

# Sends topic list to LLM
"This app covers: Annual Leave, Remote Work, Benefits...
 Generate 15 diverse test questions covering factual,
 procedural, edge case, eligibility, and comparative angles."

# LLM generates questions like:
[
  "How many vacation days do full-time employees get per year?",
  "Can contractors work remotely?",
  "What happens if a public holiday falls during annual leave?",
  "What is the maximum meal allowance for dinner during business travel?",
  ...15 total
]

# Saves to SQLite — never regenerated
```

#### Step 2 — Question Firing (Every 3 Minutes)

```python
# Picks questions in round-robin order
Cycle 1  → Question 1
Cycle 2  → Question 2
...
Cycle 15 → Question 15
Cycle 16 → Question 1 again  ← consistency testing begins
```

This design is intentional. By repeating questions, the system can detect if the app gives **different answers to the same question over time.**

---

## Agent 2 — Evaluator Agent

**File:** `agents/evaluator_agent.py`

**Purpose:** Score every answer on four metrics and detect inconsistencies across runs — with no hardcoded expected answers.

### The Four Metrics

#### 1. Faithfulness (LLM Judge)

**Question asked:** *"Is every claim in this answer actually supported by the retrieved context?"*

```
Retrieved Context: "Employees get 15 days of paid leave per year."
App Answer:        "Employees get 30 days of paid leave per year."

LLM Judge: "The answer states 30 days, but context says 15 days.
            This claim is NOT supported by the source."

Faithfulness Score: 0.05  ← WRONG ANSWER CAUGHT
```

**Why this matters:** If the app is making up facts not in the documents, faithfulness catches it immediately. This is the primary detector for hallucinations and wrong answers.

#### 2. Relevancy (LLM Judge)

**Question asked:** *"Does this answer actually address what was asked?"*

```
Question: "How many vacation days do employees get?"
Answer:   "The company has a great remote work policy..."

LLM Judge: "The answer talks about remote work, not vacation days."

Relevancy Score: 0.10  ← OFF-TOPIC ANSWER CAUGHT
```

#### 3. Completeness (LLM Judge)

**Question asked:** *"Does the answer cover all the relevant information from the context needed to fully answer the question?"*

```
Question: "What is the leave carry-forward policy?"
Context:  "Up to 5 days can be carried forward. Unused days beyond 5 lapse."
Answer:   "You can carry forward unused leave."

LLM Judge: "Answer misses the 5-day limit and the lapse rule."

Completeness Score: 0.40  ← INCOMPLETE ANSWER CAUGHT
```

#### 4. ROUGE-L (Mathematical Formula)

ROUGE-L measures text overlap between the current answer and previous answers to the same question. It uses the **Longest Common Subsequence (LCS)** algorithm.

```
Formula:
  ROUGE-L = LCS(current_answer, previous_answer) / len(previous_answer)

Example:
  Previous: "Employees receive 15 days of paid annual leave."
  Current:  "Employees are entitled to 15 days of annual leave per year."

  LCS = "Employees ... 15 days ... annual leave"
  ROUGE-L = 0.72  ← Similar enough, good consistency
```

**Why ROUGE-L matters:** Even if the LLM judge says both answers are correct, ROUGE-L detects when the phrasing has drifted significantly. Low ROUGE-L + high faithfulness = correct but inconsistent phrasing.

### Overall Score Formula

```
Overall = 0.30 × Faithfulness
        + 0.25 × Relevancy
        + 0.25 × Completeness
        + 0.20 × ROUGE-L

Score Guide:
  0.80 – 1.00  →  GOOD    (green)
  0.60 – 0.79  →  WARNING (yellow)
  0.00 – 0.59  →  POOR    (red)
```

Faithfulness has the highest weight (0.30) because factual accuracy is the most critical property of a RAG system.

### Consistency Checking

After scoring each run, the evaluator groups all answers by question and compares them against each other.

```
Question: "How many vacation days do employees get?"

Run 1 → "15 days of paid leave"
Run 2 → "15 annual leave days"
Run 3 → "30 days vacation"      ← PROBLEM
Run 4 → "15 days paid annually"

Pairwise comparison:
  Run 1 vs Run 2 → similarity: 0.92, contradicts: NO
  Run 1 vs Run 3 → similarity: 0.15, contradicts: YES ← FLAGGED
  Run 2 vs Run 3 → similarity: 0.12, contradicts: YES ← FLAGGED
  Run 2 vs Run 4 → similarity: 0.88, contradicts: NO
  ...

Consistency Score = avg(similarities) × (1 - contradiction_rate) × (1 - drift)
                  = 0.61  ← Flagged as INCONSISTENT
```

**Three consistency signals:**

| Signal | What It Measures |
|---|---|
| Consistency Score | Overall agreement across all runs |
| Contradiction Rate | Percentage of run-pairs that factually conflict |
| Drift Score | How much the first answer differs from the latest |

---

## The LLM Client — Supporting Multiple Providers

**File:** `llm_client.py`

One file handles all three LLM providers. Switch between them with a single line in `.env`.

```python
LLM_PROVIDER=groq    # Fast, free tier available
LLM_PROVIDER=azure   # Enterprise Azure OpenAI
LLM_PROVIDER=grok    # xAI Grok models
```

The client automatically retries on rate limit errors with exponential backoff, so the system never crashes due to API throttling.

---

## The Orchestrator — Putting It All Together

**File:** `orchestrator.py`

The orchestrator is the conductor. It runs on a schedule and calls each agent in sequence.

```
Every 3 minutes:
  1. Test Agent  →  fires one question, saves result to DB
  2. Evaluator   →  scores new results, runs consistency check
  3. Report      →  updates HTML report
  4. Dashboard   →  auto-refreshes from DB (separate process)
```

The first cycle runs immediately on startup so you see data within seconds, not minutes.

---

## The Dashboard — Live Visibility

**File:** `dashboard.py`

Built with Streamlit. Reads directly from the SQLite database. Auto-refreshes every 30 seconds.

### What You See

```
┌──────────────────────────────────────────────────────────┐
│  Total Runs │ Questions │ Overall │ Faith │ Relev │ Flags │
├──────────────────────────────────────────────────────────┤
│  Score Trend Line Chart (all metrics across all runs)    │
├──────────────────────────────────────────────────────────┤
│  ⚠ Consistency Alerts                                    │
│  → "How many vacation days?" — 2 contradictions found    │
│    Run 1 vs Run 3: "15 days" contradicts "30 days"       │
├──────────────────────────────────────────────────────────┤
│  Per-Question Comparison (click to expand)               │
│  ✅ "What is the leave carry-forward limit?" (4 runs)     │
│  ⚠  "How many vacation days?" (4 runs) — INCONSISTENT   │
│     Run │ Answer         │ Faith │ Relev │ Compl │ Total  │
│     1   │ 15 days...     │ 1.00  │ 1.00  │ 1.00  │ 0.80  │
│     2   │ 15 annual...   │ 1.00  │ 1.00  │ 1.00  │ 0.80  │
│     3   │ 30 days...     │ 0.05  │ 1.00  │ 1.00  │ 0.55  │ ←
│     4   │ 15 days paid.. │ 1.00  │ 1.00  │ 1.00  │ 0.80  │
├──────────────────────────────────────────────────────────┤
│  Radar Chart — Average score across all 5 dimensions     │
└──────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Component | Technology | Why This Choice |
|---|---|---|
| RAG Application | FastAPI + Uvicorn | Lightweight, fast Python REST API |
| Document Retrieval | scikit-learn TF-IDF | No model download, runs instantly |
| LLM (answers + judging) | Groq / Azure / Grok | Switchable via config |
| Scheduling | APScheduler | Simple in-process scheduler |
| Database | SQLite | Zero setup, file-based, portable |
| Dashboard | Streamlit | Python-native, production-ready UI |
| Charts | Plotly | Interactive, professional charts |
| Text Scoring | rouge-score | Standard NLP evaluation library |

---

## How to Run

### Prerequisites
- Python 3.11+
- A Groq API key (free at console.groq.com) OR Azure OpenAI credentials

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/sagu25/TESTAGENT.git
cd TESTAGENT

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env
```

Edit `.env` and fill in your Azure OpenAI credentials:

```env
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-01
```

> **Why Azure?** Azure OpenAI has no RPM/TPM rate limits at enterprise tier, making evaluation fast and reliable. Groq free tier hits limits quickly with the 4-LLM-call evaluation pipeline.

### Run the System (3 terminals)

**Terminal 1 — Start the RAG App**
```bash
python start_rag_app.py
# Runs on http://localhost:8000
```

**Terminal 2 — Start the Evaluator Pipeline**
```bash
python orchestrator.py
# Fires questions and evaluates every 3 minutes
```

**Terminal 3 — Start the Dashboard**
```bash
streamlit run dashboard.py
# Opens at http://localhost:8501
```

---

## Demonstrating a Wrong Answer

To show the system catching a wrong answer live:

1. Open `rag_app/documents.py`
2. Change `"15 days of paid annual leave"` to `"99 days of paid annual leave"`
3. Save the file
4. Wait for the next orchestrator cycle
5. Watch the dashboard — Faithfulness will drop and a contradiction alert will appear within 2 cycles

This demonstrates that the system requires **no pre-written expected answers** to catch errors. It detects problems purely by reasoning about the answer against its source context and comparing runs against each other.

---

## Key Design Decisions

### Why No Hardcoded Expected Answers?
Pre-writing expected answers is expensive, requires domain expertise, and goes stale as the app evolves. Our LLM-as-judge approach works on any RAG app without any manual labeling work.

### Why Repeat Questions?
Repeating the same question across cycles is how we detect **non-determinism** — the most dangerous class of RAG bug, where the app gives correct answers sometimes and wrong answers other times.

### Why SQLite?
For a proof-of-concept and demo, SQLite is ideal — zero infrastructure, single file, portable. The system can be upgraded to PostgreSQL for production with a one-line change in `storage.py`.

### Why TF-IDF Instead of Vector Search?
This system is designed to test RAG apps, not be a production RAG app itself. TF-IDF is fast, requires no model download or API call, and is good enough for the demo RAG app. A real deployment would use ChromaDB, Pinecone, or pgvector.

---

## What This System Is vs What It Is Not

| This System IS | This System IS NOT |
|---|---|
| A proof-of-concept evaluation framework | A production-ready enterprise platform |
| Capable of testing any RAG app via REST API | A load testing or performance testing tool |
| Self-configuring (generates its own test questions) | A replacement for human QA testers |
| Good for demos, POCs, and early-stage projects | A security or penetration testing tool |

For enterprise use, the next steps would be: PostgreSQL, Celery task queue, Slack/PagerDuty alerting, CI/CD integration, and a React dashboard.

---

## Authors

Built with Claude Code (Anthropic) + Groq (llama-3.3-70b-versatile)
