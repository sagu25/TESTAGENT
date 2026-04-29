import os
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

_client = None
_model = None


def get_client():
    global _client, _model
    if _client:
        return _client, _model

    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        _client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        _model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    elif provider == "azure":
        _client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
        _model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

    elif provider == "grok":
        _client = OpenAI(
            api_key=os.getenv("GROK_API_KEY"),
            base_url="https://api.x.ai/v1",
        )
        _model = os.getenv("GROK_MODEL", "grok-3")

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{provider}'. Use 'groq', 'azure', or 'grok'.")

    print(f"[LLMClient] Provider: {provider.upper()} | Model: {_model}")
    return _client, _model


def chat(messages: list[dict], temperature: float = 0.2) -> str:
    client, model = get_client()
    max_retries = 5
    wait = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                print(f"[LLMClient] Rate limited. Waiting {wait}s before retry {attempt+1}/{max_retries}...")
                import time
                time.sleep(wait)
                wait *= 2
            else:
                raise
    raise RuntimeError("Max retries exceeded due to rate limiting.")
