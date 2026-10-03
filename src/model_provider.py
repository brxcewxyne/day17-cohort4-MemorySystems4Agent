from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Student TODO: define the provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


_VALID_PROVIDERS = (
    "openai",
    "custom",
    "gemini",
    "anthropic",
    "ollama",
    "openrouter",
)

_ALIASES = {
    # Scaffold typo explicitly required by the lab.
    "anthorpic": "anthropic",
}


def normalize_provider(value: str) -> str:
    """Normalize a provider name to its canonical form.

    Strips whitespace, lowercases, maps known typos (e.g. `anthorpic`
    -> `anthropic`) and validates against the supported provider set.
    Pure function: no network or SDK calls.
    """

    if not isinstance(value, str):
        raise ValueError(
            f"Provider must be a string, got {type(value).__name__}. "
            f"Supported providers: {', '.join(_VALID_PROVIDERS)}."
        )
    cleaned = value.strip().lower()
    if not cleaned:
        raise ValueError(
            f"Provider must be a non-empty string. Supported providers: "
            f"{', '.join(_VALID_PROVIDERS)}."
        )
    cleaned = _ALIASES.get(cleaned, cleaned)
    if cleaned not in _VALID_PROVIDERS:
        raise ValueError(
            f"Unsupported provider {value!r} (normalized to {cleaned!r}). "
            f"Supported providers: {', '.join(_VALID_PROVIDERS)}."
        )
    return cleaned


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    Uses lazy imports so importing this module (offline mode) never
    requires provider SDKs. Only calling this function requires them.

    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenRouter`
    """

    provider = normalize_provider(config.provider)

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is required for provider 'openai'. "
                "Install it with: pip install langchain-openai"
            ) from exc
        kwargs: dict = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key is not None:
            kwargs["api_key"] = config.api_key
        if config.base_url is not None:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    if provider == "custom":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is required for provider 'custom' "
                "(OpenAI-compatible base URL). "
                "Install it with: pip install langchain-openai"
            ) from exc
        if not config.base_url:
            raise ValueError(
                "Provider 'custom' requires a base_url "
                "(set CUSTOM_BASE_URL / config.base_url)."
            )
        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
            "base_url": config.base_url,
        }
        if config.api_key is not None:
            kwargs["api_key"] = config.api_key
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                "langchain-google-genai is required for provider 'gemini'. "
                "Install it with: pip install langchain-google-genai"
            ) from exc
        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key is not None:
            kwargs["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**kwargs)

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                "langchain-anthropic is required for provider 'anthropic'. "
                "Install it with: pip install langchain-anthropic"
            ) from exc
        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key is not None:
            kwargs["api_key"] = config.api_key
        return ChatAnthropic(**kwargs)

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError(
                "langchain-ollama is required for provider 'ollama'. "
                "Install it with: pip install langchain-ollama"
            ) from exc
        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.base_url is not None:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    if provider == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter
        except ImportError as exc:
            raise ImportError(
                "langchain-openrouter is required for provider 'openrouter'. "
                "Install it with: pip install langchain-openrouter"
            ) from exc
        kwargs = {
            "model_name": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key is not None:
            kwargs["openrouter_api_key"] = config.api_key
        if config.base_url is not None:
            kwargs["openrouter_api_base"] = config.base_url
        return ChatOpenRouter(**kwargs)

    # Unreachable because normalize_provider already validates.
    raise ValueError(f"Unsupported provider {config.provider!r}.")
