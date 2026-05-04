"""
Blueverse Foundry Connector — OAuth2 Client Credentials Flow

Step 1: POST to TOKEN_URL with client_id + client_secret → get access_token
Step 2: POST to CHAT_URL with Bearer token + question → get answer
"""
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN_URL     = os.getenv("BLUEVERSE_TOKEN_URL", "")
CHAT_URL      = os.getenv("BLUEVERSE_CHAT_URL", "")
CLIENT_ID     = os.getenv("BLUEVERSE_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("BLUEVERSE_CLIENT_SECRET", "")
VERIFY_SSL    = os.getenv("BLUEVERSE_VERIFY_SSL", "true").lower() != "false"

# Configurable field names — adjust based on Blueverse API format
REQUEST_FIELD  = os.getenv("BLUEVERSE_REQUEST_FIELD",  "message")
RESPONSE_FIELD = os.getenv("BLUEVERSE_RESPONSE_FIELD", "response")

# Token cache — reuse until expiry
_token_cache = {"token": None, "expires_at": 0}


def _get_access_token() -> str:
    if _token_cache["token"] and time.time() < _token_cache["expires_at"]:
        return _token_cache["token"]

    if not all([TOKEN_URL, CLIENT_ID, CLIENT_SECRET]):
        raise ValueError(
            "Missing Blueverse credentials. Set BLUEVERSE_TOKEN_URL, "
            "BLUEVERSE_CLIENT_ID and BLUEVERSE_CLIENT_SECRET in .env"
        )

    print("[Blueverse] Fetching access token...")
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type":    "client_credentials",
            "client_id":     CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        verify=VERIFY_SSL,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    token      = data.get("access_token", "")
    expires_in = int(data.get("expires_in", 3600))

    _token_cache["token"]      = token
    _token_cache["expires_at"] = time.time() + expires_in - 60

    print(f"[Blueverse] Token obtained. Valid for {expires_in}s.")
    return token


def configure(token_url: str, chat_url: str,
              client_id: str, client_secret: str,
              verify_ssl: bool = True,
              request_field: str = "message",
              response_field: str = "response"):
    """Dynamically configure Blueverse credentials at runtime (called by MCP tool)."""
    global TOKEN_URL, CHAT_URL, CLIENT_ID, CLIENT_SECRET
    global VERIFY_SSL, REQUEST_FIELD, RESPONSE_FIELD

    TOKEN_URL      = token_url
    CHAT_URL       = chat_url
    CLIENT_ID      = client_id
    CLIENT_SECRET  = client_secret
    VERIFY_SSL     = verify_ssl
    REQUEST_FIELD  = request_field
    RESPONSE_FIELD = response_field

    # Clear token cache so new token is fetched
    _token_cache["token"]      = None
    _token_cache["expires_at"] = 0


def query(question: str) -> dict | None:
    """Send a question to Blueverse and return in standard format."""
    if not CHAT_URL:
        raise ValueError("BLUEVERSE_CHAT_URL not set.")

    try:
        token   = _get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        }
        payload  = {REQUEST_FIELD: question}
        resp     = requests.post(CHAT_URL, json=payload, headers=headers,
                                 verify=VERIFY_SSL, timeout=60)
        resp.raise_for_status()
        data     = resp.json()

        # Try common response field names
        answer = (
            data.get(RESPONSE_FIELD)
            or data.get("answer")
            or data.get("output")
            or data.get("text")
            or data.get("content")
            or str(data)
        )

        # Extract sources if Blueverse returns them
        raw_sources = data.get("sources", data.get("source_docs",
                               data.get("references", [])))
        contexts = []
        for s in raw_sources if isinstance(raw_sources, list) else []:
            if isinstance(s, dict):
                contexts.append({
                    "source": s.get("title", s.get("name", "Blueverse")),
                    "text":   s.get("content", s.get("text", str(s))),
                    "score":  float(s.get("score", 1.0)),
                })
            elif isinstance(s, str):
                contexts.append({"source": "Blueverse", "text": s, "score": 1.0})

        return {
            "question":          question,
            "answer":            answer,
            "retrieved_context": contexts,
            "sources":           [c["source"] for c in contexts] or ["Blueverse"],
        }

    except requests.HTTPError as e:
        print(f"[Blueverse] HTTP {e.response.status_code}: {e.response.text[:200]}")
        return None
    except Exception as e:
        print(f"[Blueverse] Error: {e}")
        return None


def is_online() -> bool:
    try:
        _get_access_token()
        return True
    except Exception:
        return False
