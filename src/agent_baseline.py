from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Student TODO: implement Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None
        if not self.force_offline:
            try:
                self.langchain_agent = self._maybe_build_langchain_agent()
            except Exception:
                self.langchain_agent = None

    def _session_for(self, thread_id: str) -> SessionState:
        session = self.sessions.get(thread_id)
        if session is None:
            session = SessionState()
            self.sessions[thread_id] = session
        return session

    @staticmethod
    def _history_tokens(messages: list[dict[str, str]]) -> int:
        total = 0
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "") or "")
            content = item.get("content", "")
            total += estimate_tokens(f"{role} {'' if content is None else str(content)}")
        return total

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting.

        `user_id` is accepted for API symmetry with AdvancedAgent but is
        never used as a memory key: sessions are isolated by `thread_id`.
        Uses the live path only when a live agent exists and offline is
        not forced; otherwise uses the deterministic offline path.
        """

        _ = user_id
        if not self.force_offline and self.langchain_agent is not None:
            try:
                return self._reply_live(thread_id, message)
            except Exception:
                pass
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return 0 if session is None else int(session.token_usage)

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return 0 if session is None else int(session.prompt_tokens_processed)

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    @staticmethod
    def _offline_response_text(turn_index: int, message: str) -> str:
        words = str(message).split()
        snippet = " ".join(words[:12])
        if len(words) > 12:
            snippet += "…"
        if snippet:
            return (
                f"Baseline (thread này, lượt {turn_index}): "
                f"đã nhận {len(words)} từ — “{snippet}”. "
                "Tôi chỉ nhớ trong thread hiện tại."
            )
        return (
            f"Baseline (thread này, lượt {turn_index}): "
            "đã nhận tin nhắn rỗng. Tôi chỉ nhớ trong thread hiện tại."
        )

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline behavior with thread-local memory only.

        - Session keyed by `thread_id` (never `user_id`).
        - Full history retained, never compacted.
        - No `User.md`, no profile extraction, no cross-thread recall.
        """

        session = self._session_for(thread_id)
        text = "" if message is None else str(message)
        session.messages.append({"role": "user", "content": text})
        prompt_tokens = self._history_tokens(session.messages)
        turn_index = sum(1 for m in session.messages if m.get("role") == "user")
        response = self._offline_response_text(turn_index, text)
        session.messages.append({"role": "assistant", "content": response})
        agent_tokens = estimate_tokens(response)
        session.token_usage += agent_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        """Thin live path: same thread-local accounting, real model call."""

        session = self._session_for(thread_id)
        text = "" if message is None else str(message)
        session.messages.append({"role": "user", "content": text})
        prompt_tokens = self._history_tokens(session.messages)
        result = self.langchain_agent.invoke(text)
        response = getattr(result, "content", result)
        if not isinstance(response, str):
            response = str(response)
        session.messages.append({"role": "assistant", "content": response})
        agent_tokens = estimate_tokens(response)
        session.token_usage += agent_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        """Build the live chat model without any memory add-ons.

        Uses `build_chat_model(self.config.model)` so the baseline can run
        with any supported provider. Returns None when unavailable so the
        agent safely falls back to the offline path.
        """

        try:
            return build_chat_model(self.config.model)
        except Exception:
            return None
