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
- `fastapi` + `uvicorn` — RAG app server
- `openai` — Azure / Groq / Grok client
- `apscheduler` — 2-minute trigger scheduler
- `streamlit` — Live dashboard
- `plotly` + `pandas` — Charts and data
- `scikit-learn` — TF-IDF document search
- `rouge-score` — ROUGE-L metric
- `python-dotenv` — Environment config

---

## Step 3 — Configure API Keys

Copy the example config file:

```bash
cp .env.example .env
```

Open `.env` and fill in your Azure OpenAI credentials:

```env
# Set provider to azure
LLM_PROVIDER=azure

# Your Azure OpenAI credentials
AZURE_OPENAI_API_KEY=abc123...your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-01
```

### How to Find Your Azure Credentials

```
Azure Portal → Your OpenAI Resource → Keys and Endpoint
  AZURE_OPENAI_API_KEY      = Key 1 or Key 2
  AZURE_OPENAI_ENDPOINT     = Endpoint URL

Azure Portal → Your OpenAI Resource → Model Deployments
  AZURE_OPENAI_DEPLOYMENT_NAME = Name of your deployed model
                                 (e.g. gpt-4o, gpt-4-turbo)
```

---

## Step 4 — Run the System (3 Terminals)

Open **3 separate terminal windows** in the `TESTAGENT` folder.

### Terminal 1 — Start the RAG App

```bash
python start_rag_app.py
```

Expected output:
```
INFO: Uvicorn running on http://0.0.0.0:8000
```

Verify it works:
```bash
curl http://localhost:8000/
```

### Terminal 2 — Start the Orchestrator (Test + Evaluate Pipeline)

```bash
python orchestrator.py
```

Expected output:
```
============================================================
  RAG Evaluation System
  Trigger interval: every 120 seconds
  Provider: AZURE
  RAG App: http://localhost:8000
============================================================
[Orchestrator] Running first pipeline immediately...
[TestAgent] Analyzing RAG app to generate questions...
[LLMClient] Provider: AZURE | Model: gpt-4o
[TestAgent] Generated 15 questions from app analysis.
[TestAgent] Firing question [1/15]: ...
[GoldenGen] Generating golden answer for: ...
[EvaluatorAgent] Evaluating run 1: ...
  Layer 1 Factual: 0.95
  Layer 2 Golden ROUGE-L: 0.62
  Layer 3 LLM Judge: faith=0.90 relev=0.95 compl=0.80
  [OK] Overall Score: 0.84
[ReportGenerator] Report saved.
```

### Terminal 3 — Start the Dashboard

```bash
streamlit run dashboard.py
```

Expected output:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Open your browser at **http://localhost:8501**

---

## Step 5 — Verify Everything is Working

After the first pipeline cycle completes (about 1-2 minutes), you should see:

```
✓ http://localhost:8000        → RAG App running
✓ http://localhost:8501        → Dashboard showing scores
✓ eval_results.db              → Database created with data
✓ reports/report.html          → HTML report generated
```

---

## Trigger Intervals

The system fires every 120 seconds by default. To change:

```env
# .env
TRIGGER_INTERVAL_SECONDS=60    # every 1 minute (faster demo)
TRIGGER_INTERVAL_SECONDS=120   # every 2 minutes (default)
TRIGGER_INTERVAL_SECONDS=300   # every 5 minutes (production)
```

---

## Switching Between LLM Providers

Change `LLM_PROVIDER` in `.env` and restart the orchestrator:

```env
LLM_PROVIDER=azure    # Azure OpenAI (recommended — no rate limits)
LLM_PROVIDER=groq     # Groq (free tier — has rate limits)
LLM_PROVIDER=grok     # xAI Grok
```

---

## Demonstrating a Wrong Answer (For Demos)

This shows the evaluator catching errors in real time:

**Step 1** — Open `rag_app/documents.py`

**Step 2** — Find this line in the Annual Leave Policy section:
```python
"All full-time employees are entitled to 15 days of paid annual leave"
```

**Step 3** — Change `15` to `99`:
```python
"All full-time employees are entitled to 99 days of paid annual leave"
```

**Step 4** — Save the file (RAG app auto-reloads)

**Step 5** — Wait for the next pipeline cycle

**What you will see on the dashboard:**
```
Layer 1 Factual Anchor Score: 0.10   ← "99" not in original context
Layer 2 Golden ROUGE-L:       0.15   ← answer differs from golden
Contradicts Golden:           YES    ← LLM judge flags it
Overall Score:                0.22   ← RED — system caught the error
```

**Step 6** — Revert the change to restore correct answers

---

## File Structure Reference

```
TESTAGENT/
├── rag_app/
│   ├── main.py              Start with: python start_rag_app.py
│   ├── documents.py         Edit to inject wrong answers for demo
│   └── retriever.py         TF-IDF search (do not modify)
├── agents/
│   ├── test_agent.py        Fires questions every N seconds
│   └── evaluator_agent.py   3-layer scoring engine
├── dashboard.py             streamlit run dashboard.py
├── orchestrator.py          python orchestrator.py
├── llm_client.py            Edit to add new providers
├── metrics.py               All scoring formulas
├── factual_extractor.py     Layer 1: pure code fact checking
├── golden_answer_generator.py  Layer 2: ground truth generation
├── storage.py               SQLite database operations
├── .env                     YOUR CREDENTIALS GO HERE
├── .env.example             Template (safe to commit)
├── requirements.txt         pip install -r requirements.txt
└── reports/report.html      Open in browser for static report
```

---

## Common Issues

| Problem | Cause | Fix |
|---|---|---|
| `Authentication failed` | Wrong Azure key | Check AZURE_OPENAI_API_KEY in .env |
| `Could not reach RAG app` | start_rag_app.py not running | Start Terminal 1 first |
| `No data in dashboard` | Orchestrator not run yet | Wait for first cycle to complete |
| `Port 8000 already in use` | Another process using port | Kill it or change port in start_rag_app.py |
| `Port 8501 already in use` | Another Streamlit running | `streamlit run dashboard.py --server.port 8502` |
| `Module not found` | Dependencies missing | `pip install -r requirements.txt` |

---

## Resetting the System (Fresh Start)

```bash
# Delete the database to start fresh
del eval_results.db

# The system will regenerate questions and golden answers on next run
python orchestrator.py
```
