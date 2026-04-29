import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import time
from apscheduler.schedulers.blocking import BlockingScheduler
import storage
from agents.test_agent import run as run_test_agent
from agents.evaluator_agent import run as run_evaluator_agent

INTERVAL_SECONDS = int(os.getenv("TRIGGER_INTERVAL_SECONDS", "120"))


def pipeline():
    print(f"\n{'='*60}")
    print(f"PIPELINE TRIGGERED at {__import__('datetime').datetime.utcnow().strftime('%H:%M:%S')} UTC")
    print(f"{'='*60}")
    run_test_agent()
    run_evaluator_agent()


def main():
    print("="*60)
    print("  RAG Evaluation System")
    print(f"  Trigger interval: every {INTERVAL_SECONDS} seconds")
    print(f"  Provider: {os.getenv('LLM_PROVIDER', 'grok').upper()}")
    print(f"  RAG App: {os.getenv('RAG_APP_URL', 'http://localhost:8000')}")
    print("="*60)

    storage.init_db()

    print("\n[Orchestrator] Running first pipeline immediately...")
    pipeline()

    scheduler = BlockingScheduler()
    scheduler.add_job(pipeline, "interval", seconds=INTERVAL_SECONDS)

    print(f"\n[Orchestrator] Scheduler running. Next trigger in {INTERVAL_SECONDS}s. Press Ctrl+C to stop.\n")
    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("\n[Orchestrator] Stopped by user.")


if __name__ == "__main__":
    main()
