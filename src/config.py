from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Student TODO: load environment variables and return a LabConfig.

    Pseudocode:
    1. Resolve the repo root or default to the current file parent.
    2. Optionally load values from `.env`.
    3. Create `state/` if it does not exist.
    4. Return a populated LabConfig instance.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # Load `.env` when python-dotenv is installed; never fail without it
    # and never create or modify `.env`.
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except Exception:
        pass

    def _clean(value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        return text if text else None

    def _parse_float(value: str | None, default: float) -> float:
        if value is None:
            return default
        try:
            return float(value.strip())
        except (ValueError, AttributeError):
            return default

    def _parse_int(value: str | None, default: int) -> int:
        if value is None:
            return default
        try:
            return int(value.strip())
        except (ValueError, AttributeError):
            return default

    def _credentials_for(provider: str) -> tuple[str | None, str | None]:
        """Resolve (api_key, base_url) for one normalized provider."""
        generic_key = _clean(os.getenv("LLM_API_KEY"))
        generic_url = _clean(os.getenv("LLM_BASE_URL"))
        if provider == "openai":
            return (
                _clean(os.getenv("OPENAI_API_KEY")) or generic_key,
                _clean(os.getenv("OPENAI_BASE_URL")) or generic_url,
            )
        if provider == "custom":
            return (
                _clean(os.getenv("CUSTOM_API_KEY")) or generic_key,
                _clean(os.getenv("CUSTOM_BASE_URL")) or generic_url,
            )
        if provider == "gemini":
            return (
                _clean(os.getenv("GEMINI_API_KEY"))
                or _clean(os.getenv("GOOGLE_API_KEY"))
                or generic_key,
                _clean(os.getenv("GEMINI_BASE_URL")) or generic_url,
            )
        if provider == "anthropic":
            return (
                _clean(os.getenv("ANTHROPIC_API_KEY")) or generic_key,
                _clean(os.getenv("ANTHROPIC_BASE_URL")) or generic_url,
            )
        if provider == "ollama":
            return (
                _clean(os.getenv("OLLAMA_API_KEY")) or generic_key,
                _clean(os.getenv("OLLAMA_BASE_URL"))
                or generic_url
                or "http://localhost:11434",
            )
        if provider == "openrouter":
            return (
                _clean(os.getenv("OPENROUTER_API_KEY")) or generic_key,
                _clean(os.getenv("OPENROUTER_BASE_URL"))
                or generic_url
                or "https://openrouter.ai/api/v1",
            )
        return (generic_key, generic_url)

    provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai") or "openai")
    model_name = _clean(os.getenv("LLM_MODEL")) or "gpt-4o-mini"
    temperature = _parse_float(os.getenv("LLM_TEMPERATURE"), 0.0)
    compact_threshold_tokens = _parse_int(
        os.getenv("COMPACT_THRESHOLD_TOKENS"), 3000
    )
    compact_keep_messages = _parse_int(os.getenv("COMPACT_KEEP_MESSAGES"), 8)
    if compact_threshold_tokens <= 0:
        compact_threshold_tokens = 3000
    if compact_keep_messages <= 0:
        compact_keep_messages = 8

    api_key, base_url = _credentials_for(provider)
    model = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )

    # Judge model: dedicated JUDGE_* vars when present, otherwise fall back
    # to the main model settings. Offline mode works with no key either way.
    judge_provider_raw = (
        _clean(os.getenv("JUDGE_PROVIDER"))
        or _clean(os.getenv("LLM_JUDGE_PROVIDER"))
        or provider
    )
    judge_provider = normalize_provider(judge_provider_raw or "openai")
    judge_model_name = (
        _clean(os.getenv("JUDGE_MODEL"))
        or _clean(os.getenv("LLM_JUDGE_MODEL"))
        or model_name
    )
    judge_temperature = _parse_float(
        _clean(os.getenv("JUDGE_TEMPERATURE"))
        or _clean(os.getenv("LLM_JUDGE_TEMPERATURE")),
        temperature,
    )
    if judge_provider == provider:
        judge_api_key, judge_base_url = api_key, base_url
        judge_api_override = _clean(os.getenv("JUDGE_API_KEY"))
        judge_url_override = _clean(os.getenv("JUDGE_BASE_URL"))
        if judge_api_override is not None:
            judge_api_key = judge_api_override
        if judge_url_override is not None:
            judge_base_url = judge_url_override
    else:
        judge_api_key, judge_base_url = _credentials_for(judge_provider)
        judge_api_override = _clean(os.getenv("JUDGE_API_KEY"))
        judge_url_override = _clean(os.getenv("JUDGE_BASE_URL"))
        if judge_api_override is not None:
            judge_api_key = judge_api_override
        if judge_url_override is not None:
            judge_base_url = judge_url_override
    judge_model = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=judge_temperature,
        api_key=judge_api_key,
        base_url=judge_base_url,
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        model=model,
        judge_model=judge_model,
    )
