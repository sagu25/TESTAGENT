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

This installs:

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | RAG app REST server |
| `openai` | Azure OpenAI / Groq / Grok client |
| `apscheduler` | Automated 2-minute trigger |
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
  AZURE_OPENAI_DEPLOYMENT_NAME  = Your deployed model name (e.g. gpt-4o)
```

### Optional — Add Groq as Second Judge

If you have a Groq API key (free at console.groq.com), adding it enables
**multi-judge consensus** where Azure and Groq evaluate independently:

```env
GROQ_API_KEY=gsk_your_groq_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

> **Note:** The system works with Azure alone. Groq is optional but improves evaluation reliability.

---

## Step 4 — Run the System (2 Terminals)

Open **2 terminal windows** in the `TESTAGENT` folder.

### Terminal 1 — Start the RAG App

```bash
python start_rag_app.py
```

Expected output:
```
INFO: Uvicorn running on http://0.0.0.0:8000
```

Verify it works by opening: **http://localhost:8000/docs**

### Terminal 2 — Start the Unified App

```bash
streamlit run app.py
```

Expected output:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Open your browser at **http://localhost:8501**

> The orchestrator (`orchestrator.py`) is now optional — you can trigger test cycles
> directly from the **Start Testing** page inside the app.

---

## Step 5 — Upload Your Policy Document

1. Open **http://localhost:8501**
2. Go to **📄 Documents** page
3. Click **Browse files** → select your PDF or TXT policy document
4. Enter a title (e.g. "HR Policy 2024")
5. Click **Add Document**

The system will:
- Extract text from the PDF automatically
- Reload the RAG app index
- Clear old questions so new ones are generated from your document

---

## Step 6 — Run Your First Test

1. Go to **🧪 Start Testing** page
2. You will see 15 auto-generated questions based on your document
3. Click **Run 1 Cycle** — fires one question, evaluates, saves result
4. Click **Run 5 Cycles** — runs 5 questions in sequence with progress bar
5. Go to **📊 Dashboard** to see scores, charts, and comparison tables

---

## Step 7 — View Results

### Dashboard Page (📊)
- Live score metrics — Overall, Faithfulness, Factual Anchors, Golden ROUGE-L
- Score trend chart across all runs
- Consistency alerts for questions with contradicting answers
- Per-question expandable sections showing every run side by side
- **Download Report** button — saves full HTML report to your machine

### Human Review Page (👁)
- Answers scoring below 0.60 are flagged here automatically
- Submit verdict: Correct / Partially Correct / Incorrect
- Edit the Golden Answer directly to update ground truth

### Adversarial Questions Page (⚔)
- Add your own test questions — edge cases, trick questions, out-of-scope
- These are added to the test rotation alongside auto-generated questions

### Regression Tracking Page (📈)
- Save named snapshots: "v1.0-baseline", "after-doc-update"
- Compare any two snapshots side by side with delta arrows
- Detect score regressions after document or app changes

---

## Trigger Intervals

The orchestrator fires every 120 seconds by default. Change in `.env`:

```env
TRIGGER_INTERVAL_SECONDS=60    # every 1 minute (faster demo)
TRIGGER_INTERVAL_SECONDS=120   # every 2 minutes (default)
TRIGGER_INTERVAL_SECONDS=300   # every 5 minutes (lighter load)
```

---

## Demonstrating a Wrong Answer (For Demos)

**Step 1** — Open `rag_app/documents.py`

**Step 2** — Find in Annual Leave Policy:
```python
"All full-time employees are entitled to 15 days of paid annual leave"
```

**Step 3** — Change `15` to `99` and save

**Step 4** — In the app, go to **🧪 Start Testing** → click **Run 1 Cycle**

**Step 5** — Go to **📊 Dashboard** — you will see:
```
Layer 1 Factual Anchor Score:  0.10  ← "99" not found in source context
Layer 2 Golden ROUGE-L:        0.12  ← far from reference answer
Contradicts Golden:            YES   ← multi-judge flagged it
Overall Score:                 0.22  ← RED
```

**Step 6** — Revert the change to restore correct answers

---

## File Structure Reference

```
TESTAGENT/
├── rag_app/
│   ├── main.py                 RAG API server (start_rag_app.py)
│   ├── documents.py            Default policy docs (edit for demo)
│   ├── retriever.py            TF-IDF search engine
│   └── document_store.py       Dynamic doc management (upload/remove)
│
├── agents/
│   ├── test_agent.py           Auto-generates questions, fires them
│   └── evaluator_agent.py      3-layer + multi-judge scoring engine
│
├── app.py                      ← MAIN APP (streamlit run app.py)
├── orchestrator.py             Auto-scheduler (optional, runs every 2 min)
├── multi_judge.py              Azure + Groq consensus evaluation
├── llm_client.py               Provider switcher (Azure/Groq/Grok)
├── metrics.py                  All scoring formulas
├── factual_extractor.py        Layer 1: pure code fact checking
├── golden_answer_generator.py  Layer 2: ground truth from full docs
├── storage.py                  SQLite database (all tables)
├── report_generator.py         HTML report builder
├── start_rag_app.py            Starts RAG server on port 8000
├── uploads/                    Uploaded documents stored here
├── reports/report.html         Latest HTML report
├── .env                        ← YOUR CREDENTIALS (never commit this)
├── .env.example                Template (safe to share)
└── requirements.txt            pip install -r requirements.txt
```

---

## Common Issues and Fixes

| Error | Cause | Fix |
|---|---|---|
| `Missing credentials` | Azure keys not set in .env | Fill in `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` |
| `Authentication failed` | Wrong Azure key or endpoint | Copy fresh key from Azure Portal → Keys and Endpoint |
| `RAG App: Offline` (red in sidebar) | `start_rag_app.py` not running | Open Terminal 1 and run `python start_rag_app.py` |
| `No data in dashboard` | No test cycles run yet | Go to Start Testing → Run 1 Cycle |
| `Port 8000 already in use` | Another process on port 8000 | Kill it or change port in `start_rag_app.py` |
| `Port 8501 already in use` | Another Streamlit running | `streamlit run app.py --server.port 8502` |
| `Module not found` | Dependencies not installed | `pip install -r requirements.txt` |
| `DeprecationWarning: utcnow` | Python 3.12 warning | Already fixed in latest code — pull latest from git |
| `Rate limit exceeded` (Groq) | Groq free tier limit hit | Switch to Azure: set `LLM_PROVIDER=azure` in .env |
| `PDF not reading correctly` | Scanned/image PDF | Convert to text-based PDF or copy text to a .txt file |

---

## Switching Between LLM Providers

Change `LLM_PROVIDER` in `.env` and restart both terminals:

```env
LLM_PROVIDER=azure    # Recommended — enterprise, no rate limits
LLM_PROVIDER=groq     # Free tier — 30 RPM limit, slower evaluation
LLM_PROVIDER=grok     # xAI Grok models
```

---

## Multi-Judge Setup (Recommended)

For maximum evaluation reliability, configure both Azure AND Groq.
The system will use both as independent judges and flag disagreements:

```env
LLM_PROVIDER=azure

# Primary judge
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-01

# Secondary judge (optional but recommended)
GROQ_API_KEY=gsk_your_groq_key
GROQ_MODEL=llama-3.1-8b-instant
```

When both are configured:
```
[Multi-Judge (2 models)]: faith=0.90 relev=0.95 compl=0.85
vs
[Single Judge]: faith=0.90 relev=0.95 compl=0.85

If judges disagree: [DISPUTED] disagreement=0.35 → flag for human review
```

---

## Resetting the System (Fresh Start)

```bash
# Windows
del eval_results.db

# Mac/Linux
rm eval_results.db

# System will regenerate questions and golden answers on next run
streamlit run app.py
```

To also clear uploaded documents:
```bash
# Windows
del uploads\*.txt
del uploads\meta.json
```
