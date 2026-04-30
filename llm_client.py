import os
import time
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

_client = None
_model  = None


def get_client():
    global _client, _model
    if _client:
        return _client, _model

    provider = os.getenv("LLM_PROVIDER", "azure").lower()

    if provider == "azure":
        api_key    = os.getenv("AZURE_OPENAI_API_KEY", "")
        endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        api_version= os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

        if not api_key or api_key == "your_azure_api_key_here":
            raise ValueError(
                "[LLMClient] Azure API key not set. "
                "Add AZURE_OPENAI_API_KEY to your .env file."
            )
        if not endpoint or endpoint == "https://your-resource.openai.azure.com/":
            raise ValueError(
                "[LLMClient] Azure endpoint not set. "
                "Add AZURE_OPENAI_ENDPOINT to your .env file."
            )

        _client = AzureOpenAI(
            api_key        = api_key,
            azure_endpoint = endpoint,
            api_version    = api_version,
        )
        _model = deployment

    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key or api_key == "your_groq_api_key_here":
            raise ValueError("[LLMClient] Groq API key not set. Add GROQ_API_KEY to your .env file.")
        _client = OpenAI(
            api_key  = api_key,
            base_url = "https://api.groq.com/openai/v1",
        )
        _model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    elif provider == "grok":
        api_key = os.getenv("GROK_API_KEY", "")
        if not api_key or api_key == "your_grok_api_key_here":
            raise ValueError("[LLMClient] Grok API key not set. Add GROK_API_KEY to your .env file.")
        _client = OpenAI(
            api_key  = api_key,
            base_url = "https://api.x.ai/v1",
        )
        _model = os.getenv("GROK_MODEL", "grok-3")

    else:
        raise ValueError(
            f"[LLMClient] Unknown LLM_PROVIDER: '{provider}'. "
            "Use 'azure', 'groq', or 'grok' in your .env file."
        )

    print(f"[LLMClient] Provider: {provider.upper()} | Model: {_model}")
    return _client, _model


def chat(messages: list[dict], temperature: float = 0.2) -> str:
    client, model = get_client()
    max_retries = 6
    wait = 10
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model       = model,
                messages    = messages,
                temperature = temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err or "too many" in err:
                print(f"[LLMClient] Rate limited. Waiting {wait}s (retry {attempt+1}/{max_retries})...")
                time.sleep(wait)
                wait = min(wait * 2, 120)
            elif "authentication" in err or "401" in err or "403" in err:
                raise ValueError(
                    f"[LLMClient] Authentication failed. Check your API key and endpoint in .env. Error: {e}"
                )
            else:
                raise
    raise RuntimeError("[LLMClient] Max retries exceeded. Check your API quota.")
