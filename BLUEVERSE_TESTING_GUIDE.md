# Blueverse Agent Testing Guide

How to test a Blueverse RAG-backed agent using the RAG Evaluation System,
and how to register the MCP server with Blueverse Foundry.

---

## Overview

```
What we are testing:
  A Blueverse agent that has a RAG tool connected to
  the employee policy document (Leave, Benefits, Remote Work, etc.)

How it works:
  Our MCP Server fires questions at the Blueverse agent every 30 seconds
  Evaluates each answer across 3 layers (factual, mathematical, AI judge)
  Checks consistency across multiple runs of the same question
  Shows live results on the Streamlit dashboard
```

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │     RAG Evaluation System        │
                    │                                  │
  [Start Auto]  ──► │  Test Agent                      │
  button clicks     │    picks question every 30s      │
                    │    └──► OAuth2 token              │
                    │    └──► POST question to          │
                    │         Blueverse Chat URL        │
                    │                  │                │
                    │         ◄── answer from           │
                    │              Blueverse            │
                    │                  │                │
                    │  Evaluator Agent                  │
                    │    Layer 1: Factual check         │
                    │    Layer 2: Golden ROUGE-L        │
                    │    Layer 3: LLM Judge             │
                    │    Consistency across runs        │
                    │                  │                │
                    │  Dashboard shows live scores      │
                    └─────────────────────────────────┘
                                    │
              ┌─────────────────────┴──────────────────────┐
              │          Blueverse Foundry                  │
              │                                            │
              │  [Your RAG Agent]                          │
              │    ├── Employee Policy RAG Tool            │
              │    │   (Leave, Benefits, Remote Work...)   │
              │    └── Receives questions from evaluator   │
              │         answers using document retrieval   │
              └─────────────────────────────────────────────┘
```

---

## Part 1 — Get Blueverse API Credentials

Before starting, collect these 4 values from your Blueverse admin or documentation:

| Value | Where to Find | Example |
|---|---|---|
| `BLUEVERSE_TOKEN_URL` | Blueverse documentation / admin | `https://login.microsoftonline.com/.../oauth2/token` |
| `BLUEVERSE_CHAT_URL` | Blueverse agent API docs | `https://blueverse-foundry.ltimindtree.com/api/v1/agents/abc/chat` |
| `BLUEVERSE_CLIENT_ID` | Blueverse app registration | `abc-123-def-456` |
| `BLUEVERSE_CLIENT_SECRET` | Blueverse app registration | `your_secret_here` |

Also find out:
- **Request field name**: what JSON key holds the question (`message`, `query`, `input`, `prompt`)
- **Response field name**: what JSON key holds the answer (`response`, `answer`, `output`, `text`)

---

## Part 2 — Option A: Test via Streamlit Dashboard

This is the simplest option. No MCP needed.

### Step 1 — Fill in `.env`

Open `eval_system/.env` and update:

```env
# Switch to Blueverse mode
RAG_APP_URL=blueverse

# Blueverse credentials
BLUEVERSE_TOKEN_URL=https://your-token-endpoint/oauth2/token
BLUEVERSE_CHAT_URL=https://blueverse-foundry.ltimindtree.com/api/v1/your-agent/chat
BLUEVERSE_CLIENT_ID=your_client_id
BLUEVERSE_CLIENT_SECRET=your_client_secret
BLUEVERSE_VERIFY_SSL=true

# What the agent covers (used to generate test questions)
BLUEVERSE_DESCRIPTION=LTIMindtree employee policy assistant covering annual leave,
remote work policy, code of conduct, employee benefits,
performance review, and expense reimbursement

# Field names — adjust based on Blueverse API format
BLUEVERSE_REQUEST_FIELD=message
BLUEVERSE_RESPONSE_FIELD=response
```

### Step 2 — Start the Evaluation System

```bash
# Terminal 1 — Start Streamlit app
streamlit run app.py

# Note: You do NOT need to start start_rag_app.py
# because we are testing Blueverse, not our mock RAG app
```

Make sure Cisco VPN is connected.

### Step 3 — Go to Start Testing Page

```
Open browser → http://localhost:8501
Navigate to: 🧪 Start Testing

You will see the Auto Testing section at the top.

Click: ▶ Start Auto
Set interval: 30 seconds

What happens:
  - System reads BLUEVERSE_DESCRIPTION from .env
  - Generates 15 grounded test questions about employee policies
  - Fires 1 question every 30 seconds to your Blueverse agent
  - Gets OAuth2 token automatically before each call
  - Evaluates each answer across all 3 layers
  - Dashboard updates every 30 seconds
```

### Step 4 — View Results

```
Navigate to: 📊 Dashboard

You will see:
  - Live scores updating every 30 seconds
  - Per-question comparison showing all runs
  - Consistency alerts if answers differ across runs
  - Download Report button for HTML report

After 5+ minutes: all 15 questions asked at least once
After 10+ minutes: questions repeat → consistency data appears
```

---

## Part 2 — Option B: Test via MCP Server (Blueverse controls the testing)

This option allows Blueverse itself to trigger and control the evaluation.

### Step 1 — Start the MCP Server

```bash
# In eval_system folder
python mcp_server.py

# Expected output:
# Starting RAG Evaluator MCP Server on port 8502...
# Register this URL in Blueverse: http://localhost:8502/sse
```

### Step 2 — Expose MCP Server Using ngrok

The MCP server runs locally. Blueverse (cloud service) needs a public URL.
ngrok creates a secure tunnel from public internet to your laptop.

```bash
# Install ngrok if not installed
# Download from: https://ngrok.com/download

# Start ngrok tunnel on port 8502
ngrok http 8502

# Expected output:
# Forwarding  https://abc123.ngrok-free.app → http://localhost:8502
#
# Copy the https URL — this is your MCP server public address
```

**Important:** Every time you restart ngrok, you get a new URL.
Keep ngrok running for the entire testing session.

### Step 3 — Register MCP Server in Blueverse Foundry

```
1. Open Blueverse Foundry
   → https://blueverse-foundry.ltimindtree.com

2. Go to your Agent configuration

3. Find "MCP Tools" or "External Tools" or "Tool Servers" section

4. Add new MCP server:
   Name:     RAG Evaluator
   URL:      https://abc123.ngrok-free.app/sse
             (use the ngrok URL from Step 2)
   Type:     SSE (Server-Sent Events)

5. Save the configuration

6. Blueverse will now discover 5 tools:
   - configure_agent
   - start_testing
   - stop_testing
   - get_status
   - get_latest_results
```

### Step 4 — Trigger Testing from Blueverse

Talk to your Blueverse agent and say:

**Configure the evaluator:**
```
"Configure the RAG evaluator with these Blueverse credentials:
 token_url: https://your-token-endpoint/oauth2/token
 chat_url: https://blueverse-foundry.ltimindtree.com/api/v1/your-agent/chat
 client_id: your_client_id
 client_secret: your_client_secret
 description: Employee policy assistant covering leave, benefits, remote work"
```

Blueverse calls: `configure_agent(token_url=..., chat_url=..., ...)`

**Start testing:**
```
"Start testing the Blueverse agent every 30 seconds"
```

Blueverse calls: `start_testing(interval_seconds=30)`

Response from MCP tool:
```
Testing started.
Interval:    Every 30 seconds
Target:      Blueverse agent at https://...
First cycle: Running now

Call get_status() to see live scores.
Call stop_testing() to stop.
```

**Check scores:**
```
"What is the current evaluation score?"
```

Blueverse calls: `get_status()`

Response:
```
=== RAG Evaluator Status ===
Testing running:      YES
Interval:             30s
Runs this session:    8
Total runs in DB:     8

=== Scores ===
Overall Score:        0.87 (GOOD)
Faithfulness:         0.91
Factual Anchors:      1.00
Flagged Inconsistent: 0
Contradicts Golden:   0
```

**Get detailed results:**
```
"Show me the latest evaluation results"
```

Blueverse calls: `get_latest_results(last_n=5)`

**Stop testing:**
```
"Stop testing"
```

Blueverse calls: `stop_testing()`

---

## What Gets Evaluated

For every answer the Blueverse agent gives, the system scores it across:

### Layer 1 — Factual Anchor Score (Zero AI)
```
Extracts numbers, percentages, dollar amounts from source policy documents.
Checks if the same facts appear in the Blueverse answer.

Example:
  Policy says: "15 days annual leave"
  Blueverse says: "15 days paid leave" → Score: 1.00
  Blueverse says: "30 days paid leave" → Score: 0.00 (flagged immediately)
```

### Layer 2 — Golden ROUGE-L (Pure Math)
```
Generates a perfect reference answer from the full policy document.
Measures text similarity using the ROUGE-L formula (Longest Common Subsequence).

Score 1.00 = answer is very close to the reference
Score 0.00 = answer shares almost nothing with reference
```

### Layer 3 — LLM Judge (Grounded by Reference)
```
AI judge evaluates three dimensions:
  Faithfulness:  Is every claim grounded in the source document?
  Relevancy:     Does the answer address the question asked?
  Completeness:  Is any important information missing?

The judge has the golden reference answer as a benchmark —
making it much harder for the judge itself to be wrong.
```

### Consistency Check (Cross-Run)
```
When the same question is asked multiple times:
  Compares all answers pairwise
  Detects contradictions (e.g., "15 days" vs "30 days")
  Calculates consistency score (0.0 to 1.0)
  Flags if below 0.75

Shows: is Blueverse giving stable, reliable answers over time?
```

---

## Understanding the Scores

| Score | Color | Meaning | Action |
|---|---|---|---|
| 0.80 – 1.00 | Green | Blueverse answering correctly | No action needed |
| 0.60 – 0.79 | Yellow | Some issues detected | Investigate flagged questions |
| 0.00 – 0.59 | Red | Significant problems | Check Blueverse agent configuration |

| Consistency | Meaning |
|---|---|
| 1.00 | Perfect — same answer every time |
| 0.75 – 0.99 | Minor phrasing variation — acceptable |
| Below 0.75 | FLAGGED — Blueverse giving different answers |

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| 403 error on Blueverse calls | VPN not connected | Connect Cisco VPN first |
| Token fetch fails | Wrong client_id or secret | Check credentials with Blueverse admin |
| Empty answers from Blueverse | Wrong CHAT_URL | Verify the agent endpoint URL |
| Wrong response field | Blueverse uses different field name | Set `BLUEVERSE_RESPONSE_FIELD=answer` (or correct name) |
| MCP tools not showing in Blueverse | ngrok URL wrong or MCP server not running | Check ngrok is running, verify /sse URL |
| ngrok URL expired | ngrok session ended | Restart ngrok, update URL in Blueverse |
| Questions unrelated to employee policy | BLUEVERSE_DESCRIPTION not set | Fill in description in .env |

---

## Quick Reference — Commands

```bash
# Test via dashboard (simplest)
streamlit run app.py
# Go to Start Testing → click Start Auto

# Test via MCP server
python mcp_server.py           # start MCP server
ngrok http 8502                # expose via ngrok
# Register ngrok URL in Blueverse → use tools

# Check if Blueverse connection works
python -c "
from dotenv import load_dotenv; load_dotenv()
import blueverse_connector
result = blueverse_connector.query('How many vacation days do employees get?')
print('Answer:', result['answer'] if result else 'FAILED')
"

# Reset database for fresh start
del eval_results.db
```

---

## File Reference

| File | Purpose |
|---|---|
| `blueverse_connector.py` | OAuth2 auth + API call to Blueverse |
| `mcp_server.py` | MCP server exposing 5 evaluation tools |
| `app.py` | Streamlit dashboard with Auto Testing button |
| `.env` | Credentials — fill in Blueverse values here |
| `BLUEVERSE_TESTING_GUIDE.md` | This guide |
