"""
LLM model registry with pre-initialized instances.

LLM Registery: the list of models. 
It create each model once, with its settings (name, reasoning effort, token limit).
It never calls the API itself.

Think of it as the contacts list in our phone: everyone saved and ready to dial.

ChatOpenAI sends requests to OpenAI's server (https://api.openai.com/v1) with our OpenAI key.

# Default: OpenAI's server, your OpenAI key
_build_llm("gpt-5.6-luna")

# Self-hosted vLLM
_build_llm("openai:my-model", base_url="http://localhost:8000/v1", api_key="EMPTY")  # vLLM has no key by default; any placeholder works

# OpenRouter
_build_llm("openai:some/model", base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)

Why they must go together:
Each key belongs to one server. If you change base_url but keep your OpenAI key, that key gets sent to someone elses server, 
which may log it. The guard in _build_llm blocks exactly that.

Think of base_url as a building address and api_key as the keycard for that building.

https://api.openai.com/v1     + /chat/completions  → OpenAI's server
http://localhost:8000/v1      + /chat/completions  → your vLLM server
https://openrouter.ai/api/v1  + /chat/completions  → OpenRouter's server

So changing base_url sends the request to a different machine entirely, and that’s why the key has to change with it.
"""

from typing import List, Dict, Any

from src.core.config import get_settings

from pydantic import SecretStr

from langchain.chat_models import init_chat_model

settings = get_settings()

_API_KEY = SecretStr(settings.openai_api_key)


def _is_openai_model(model:str) -> bool:
    """
    Return whether 'model' is from OpenAI provider.

    If the model start with gpt-* name (no provider prefix) is OpenAI or if it start with 'openai:' prefix is also OpenAI
    """
    return model.startswith("openai:") or (":" not in model and model.startswith("gpt-"))

# Create a LangChain chat model with a safety check for OpenAI-compatible endpoints
# kwargs: a dictionary of extra arguments passed in
def _build_llm(model: str, **kwargs: Any):

    """
    Construct a chat model via 'init_chat_model'.

    'model' accepts either a name ("gpt-5.6-luna") or an explicit "provider:model" string for any other provider (e.g. "anthropic:claude-opus-4-6", "ollama: llama3")

    Non-OpenAI providers need their LangChain integration
    package installed (see the `anthropic` / `ollama` extras in
    pyproject.toml) and read their API key from the provider's own env var
    (e.g. `ANTHROPIC_API_KEY`). An OpenAI-compatible third party (an
    `openai:` model with a custom `base_url`) must pass its own `api_key`.

    1. Takes a model string. It can be a plain name like "gpt-5.6-luna" (treated as OpenAI), or "provider:model" like "anthropic:claude-opus-4-6" for other providers.

    2. If its an OpenAI model (_is_openai_model(model), defined elsewhere), it runs two steps:
        A) Safety check
        B) Default key: kwargs.setdefault("api_key", _API_KEY) uses your projects key only if you didnt supply one.

    3. Builds the model with init_chat_model(model, **kwargs), which picks the right LangChain class for the provider.
    """
    if _is_openai_model(model):

        # Without its own key, ChatOpenAI would send OPENAI_API_KEY to the custom endpoint.
        # Check by key names, not value, so api_key=None passed the check
        custom_url = kwargs.get("base_url") or kwargs.get("openai_api_base") # base_url ~ where the request goes (the server address) - it picks which server the request goes to
        own_key = kwargs.get("api_key") or kwargs.get("openai_api_key") # Who you are on that server

        if custom_url and not own_key:
            raise ValueError(f" '{model}' sets a custom base_url but not api_key; pass that endpoint's own api_key")


        # Mistake: setdefault added api_key even when openai_api_key was passed
        # and api_key overrides it, so _API_KEY could reach the custom endpoint
        if not own_key:
            kwargs["api_key"] = _API_KEY


    return init_chat_model(model, **kwargs)



class LLMRegistery:
    """
    Registry of available LLM models with pre-initialized instances.

    This class maintains a list of LLM configurations and provides methods to retrieve them
    by name with optional argument overrides.

    """

    # 'model' is passed straight to 'init_chat_model', so an entry for another provider only needs a "provider:model" string here
    # e.g,   {"name":"claude-opus", "model": "anthropic:claude-opus-4.6", "kwargs": {...}}


    LLM_CONFIGS: list[Dict[str, Any]] = [

     {
            "name": "gpt-5.6-luna",
            "model": "gpt-5.6-luna",
            "kwargs": {
                "max_completion_tokens": settings.max_tokens,
                "reasoning": {"effort": "medium"},
            },
        },
        {
            "name": "gpt-5.4",
            "model": "gpt-5.4",
            "kwargs": {
                "max_completion_tokens": settings.max_tokens,
                "reasoning": {"effort": "medium"},
            },
        },
        {
            "name": "gpt-5.4-mini",
            "model": "gpt-5.4-mini",
            "kwargs": {
                "max_completion_tokens": settings.max_tokens,
                "reasoning": {"effort": "low"},
            },
        },
        {
            "name": "gpt-5.4-nano",
            "model": "gpt-5.4-nano",
            "kwargs": {
                "max_completion_tokens": settings.max_tokens,
                "reasoning": {"effort": "low"},
            },
        },

    ]

    LLMS: List[Dict[str, Any]] = [
        {"name": config["name"], "llm": _build_llm(config["model"], **config.get("kwargs", {}))}
        for config in LLM_CONFIGS
    ]

# next takes two arguments: 1) iterator 2) optional default argument. next() it is implementation of __next__ dunder method.
# next(iter([...])) --> return first item in the list, then second run we get the 2nd item in the list. Until end of iteration

    @classmethod
    def get(cls, model_name:str, **kwargs):
        """

        """
        # 
        # Iterator is an object that give us the tuple values one at a time - So printing shows the object and not the values
        # To get values, loop over it -> then call next() or convert it into list
        config = next((c for c in cls.LLM_CONFIGS if c["name"]==model_name), None) # We get the configuration of the selected model

        if not config:
            available = ",".join(c["name"] for c in cls.LLM_CONFIGS)
            raise ValueError(f"model {model_name} not found in registery. Available models: {available}")

        




if __name__ == "__main__":
    llm = LLMRegistery()