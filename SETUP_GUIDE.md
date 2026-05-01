# Setup Guide — RAG Test & Evaluator Agent System

Complete step-by-step instructions to get the system running from scratch.

---

## Prerequisites

| Requirement | Version | Check Command |
|---|---|---|
| Python | 3.11 or higher | `python --version` |
| pip | Latest | `pip --version` |
| Azure OpenAI | Active deployment | Azure Portal |
| Git | Any | `git --version` |

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/sagu25/TESTAGENT.git
cd TESTAGENT
```

---

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | RAG app REST server |
| `openai` | Azure OpenAI / Groq / Grok client |
| `aiohttp` | Async parallel HTTP firing |
| `tenacity` | Retry logic with exponential backoff |
| `apscheduler` | Automated scheduled trigger |
| `streamlit` | Unified UI app |
| `plotly` + `pandas` | Charts and data tables |
| `scikit-learn` | TF-IDF document retrieval |
| `rouge-score` | ROUGE-L mathematical metric |
| `pypdf` | PDF document parsing |
| `python-dotenv` | Environment config loader |

---

## Step 3 — Configure API Keys

Copy the template:

```bash
cp .env.example .env
```

Open `.env` and fill in your **Azure OpenAI** credentials:

```env
LLM_PROVIDER=azure

AZURE_OPENAI_API_KEY=your_actual_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-01
```

### Where to Find Azure Credentials

```
Azure Portal → Your OpenAI Resource → Keys and Endpoint
  AZURE_OPENAI_API_KEY          = Key 1 or Key 2
  AZURE_OPENAI_ENDPOINT         = Endpoint URL

Azure Portal → Your OpenAI Resource → Model Deployments
  AZURE_OPENAI_DEPLOYMENT_NAME  = Your deployed model (e.g. gpt-4o)
```

### Optional — Add Groq as Second Judge

Adding a Groq API key (free at console.groq.com) enables **multi-judge consensus**:

```env
GROQ_API_KEY=gsk_your_groq_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

When both are configured, Azure and Groq judge independently and scores are averaged. Disputes flagged when they disagree by more than 0.25.

---

## Step 4 — Run the System (2 Terminals)

### Terminal 1 — Start the RAG App

```bash
python start_rag_app.py
```

Expected output:
```
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Terminal 2 — Start the Unified App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

> **Sidebar check:** The sidebar shows **"RAG App: Online"** in green when both are running correctly.

---

## Step 5 — Upload Your Policy Document

1. Go to **📄 Documents** page
2. Click **Browse files** → select your PDF or TXT file
3. Enter a title (e.g. "HR Policy 2024")
4. Click **Add Document**

The system will extract text, reload the RAG index, and clear old questions so new ones are generated from your document on the next test run.

---

## Step 6 — Run Your First Test

1. Go to **🧪 Start Testing**
2. You will see 15 auto-generated questions based on your documents
3. Click **Run 1 Cycle** — fires 3 questions in parallel, evaluates all
4. Click **Run 5 Cycles** — runs 5 batches (15 questions total)
5. Go to **📊 Dashboard** to see live results

---

## Step 7 — Understand What You See

### Dashboard (📊)

```
Top Row — Grounded Metrics (Zero LLM):
  Factual Anchor Score   Are the numbers in the answer correct?
  Golden ROUGE-L         Text overlap with ground truth answer
  Contradicts Golden     Runs where answer contradicts reference
  Inconsistent Questions Questions with contradicting answers across runs

Second Row — LLM Judge Metrics:
  Overall Score          Weighted combination of all layers
  Faithfulness           Answer grounded in source context?
  Relevancy              Answer addresses the question?
  Completeness           Any important info missing?

New — Retrieval Metrics (Layer 0):
  Context Precision      % of retrieved chunks that are relevant
  Context Recall         Does retrieval contain enough info to answer?
```

### Per-Question Section
- Click any question to expand all runs side by side
- **Golden Reference Answer** shown at top (ground truth from full documents)
- Each run shows all 7 scores
- Red warnings highlight contradictions and hallucinated facts

---

## Environment Variables Reference

Full `.env` options:

```env
# Provider selection
LLM_PROVIDER=azure                    # azure | groq | grok

# Azure OpenAI (primary — recommended)
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-01

# Groq (optional second judge)
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.1-8b-instant

# Grok / xAI (alternative)
GROK_API_KEY=...
GROK_MODEL=grok-3

# RAG App
RAG_APP_URL=http://localhost:8000

# Scheduling
TRIGGER_INTERVAL_SECONDS=120         # how often orchestrator fires (seconds)
QUESTIONS_PER_CYCLE=3                # how many questions fired in parallel per cycle
```

---

## Demonstrating a Wrong Answer (For Demos)

**Step 1** — Open `rag_app/documents.py`

**Step 2** — Find in Annual Leave Policy:
```python
"All full-time employees are entitled to 15 days of paid annual leave"
```

**Step 3** — Change `15` to `99` and save

**Step 4** — Go to **🧪 Start Testing** → click **Run 1 Cycle**

**Step 5** — Go to **📊 Dashboard** — you will see:
```
Factual Anchor Score:  0.10  ← "99" not in source context
Golden ROUGE-L:        0.12  ← far from reference answer
Contradicts Golden:    YES
Overall Score:         0.22  ← RED — caught immediately
```

**Step 6** — Revert `99` back to `15`

---

## Testing Each Feature

### Human Review (👁)
```
1. Run several cycles so some scores fall below 0.60
2. Go to Human Review page
3. Low-scoring answers appear automatically
4. Submit verdict: Correct / Partially Correct / Incorrect
5. Optionally edit the Golden Answer to improve ground truth
6. Completed reviews saved and visible in table below
```

### Adversarial Questions (⚔)
```
1. Go to Adversarial Questions page
2. Add a question, select type (adversarial / edge case / trick / out of scope)
3. Example adversarial: "Can I take 200 days of annual leave?"
4. Example out of scope: "What is the company share price?"
5. Go to Start Testing — your questions appear in the list
6. They get fired automatically in the test rotation
```

### Regression Tracking (📈)
```
1. Run 5+ cycles, then go to Regression page
2. Click "Save Snapshot" → name it "baseline"
3. Upload a new/changed document or run more cycles
4. Save another snapshot → "after-change"
5. Compare the two — delta arrows show regressions
6. Score history chart shows trend across all snapshots
```

---

## Production Features Active

| Feature | How It Works |
|---|---|
| **Parallel firing** | 3 questions fired simultaneously via asyncio + aiohttp |
| **Dead letter queue** | Failed questions auto-retried next cycle (max 3 attempts) |
| **Smart prioritization** | Low-scoring and never-asked questions tested more often |
| **LLM caching** | Same question+answer = cached score, zero extra LLM cost |
| **Parallel judges** | Faith + Relevancy + Completeness evaluated simultaneously |
| **Eval versioning** | Every score tagged with prompt hash — versions never mixed |
| **Retrieval metrics** | Context precision + recall per run |
| **Multi-judge** | Azure + Groq judge independently, disputes flagged |

---

## File Structure Reference

```
TESTAGENT/
│
├── rag_app/
│   ├── main.py                  RAG API (start with start_rag_app.py)
│   ├── documents.py             Default policy docs — edit for demo
│   ├── retriever.py             TF-IDF search + dynamic reload
│   └── document_store.py        Upload/remove documents dynamically
│
├── agents/
│   ├── test_agent.py            Async parallel firing + DLQ + prioritization
│   └── evaluator_agent.py       4-layer + parallel judges + caching + retrieval
│
├── app.py                       ← MAIN APP  →  streamlit run app.py
├── orchestrator.py              Auto-scheduler (optional — app has manual trigger)
│
├── multi_judge.py               Azure + Groq consensus evaluation
├── cache.py                     LLM response cache (SQLite backed)
├── eval_version.py              Prompt hash versioning
├── retrieval_metrics.py         Context precision + recall (Layer 0)
├── llm_client.py                Provider switcher with retry logic
├── metrics.py                   All scoring formulas
├── factual_extractor.py         Layer 1: pure code fact checking
├── golden_answer_generator.py   Layer 2: ground truth from full documents
├── storage.py                   All SQLite operations
├── report_generator.py          Static HTML report builder
├── start_rag_app.py             Starts RAG server on port 8000
│
├── uploads/                     User-uploaded documents stored here
├── reports/report.html          Latest generated HTML report
│
├── .env                         ← YOUR CREDENTIALS — never commit this
├── .env.example                 Safe template to share
└── requirements.txt             pip install -r requirements.txt
```

---

## Common Issues and Fixes

| Error / Symptom | Cause | Fix |
|---|---|---|
| Sidebar shows "RAG App: Offline" | `start_rag_app.py` not running | Open Terminal 1 → `python start_rag_app.py` |
| `Missing credentials` error | Azure keys not set in `.env` | Fill `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` |
| `Authentication failed` | Wrong key or endpoint | Get fresh key from Azure Portal → Keys and Endpoint |
| No questions on Start Testing page | First run or questions were reset | Click Run 1 Cycle — questions auto-generate |
| Dashboard shows no data | No cycles run yet | Go to Start Testing → Run 1 Cycle |
| `Rate limit exceeded` on Groq | Free tier TPM/RPM limit hit | Switch to `LLM_PROVIDER=azure` in `.env` |
| `DeprecationWarning: utcnow` | Python 3.12 deprecation | Already fixed — pull latest: `git pull origin main` |
| `Port 8000 already in use` | Another process on that port | Kill it or change port in `start_rag_app.py` |
| `Port 8501 already in use` | Another Streamlit running | `streamlit run app.py --server.port 8502` |
| `Module not found` | Missing dependency | `pip install -r requirements.txt` |
| PDF not extracting text | Scanned/image PDF | Convert to text PDF or paste content into a `.txt` file |
| Score shows N/A | Column missing in old DB | Delete `eval_results.db` and rerun to rebuild schema |

---

## Switching LLM Providers

Change `LLM_PROVIDER` in `.env` and restart both terminals:

```env
LLM_PROVIDER=azure    # Recommended — enterprise, no rate limits
LLM_PROVIDER=groq     # Free tier — 30 RPM limit, use for secondary judge only
LLM_PROVIDER=grok     # xAI Grok models
```

---

## Resetting the System

```bash
# Reset database (Windows)
del eval_results.db

# Reset database (Mac/Linux)
rm eval_results.db

# Reset uploaded documents (Windows)
del uploads\*.txt
del uploads\meta.json
```

After reset, restart both terminals. The system rebuilds the database, regenerates questions, and generates new golden answers on the first run.

---

## Optional — Run the Orchestrator (Auto Mode)

Instead of clicking "Run Cycles" manually, the orchestrator fires automatically:

```bash
python orchestrator.py
```

Configurable via `.env`:
```env
TRIGGER_INTERVAL_SECONDS=120    # fire every 2 minutes
QUESTIONS_PER_CYCLE=3           # 3 questions per cycle (parallel)
```
