from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


_PROFILE_KEY_ORDER = (
    "name",
    "location",
    "job",
    "style",
    "drink",
    "food",
    "pet",
    "interests",
)

_PROFILE_LABELS = {
    "name": "Tên",
    "location": "Nơi ở",
    "job": "Nghề nghiệp",
    "style": "Style trả lời",
    "drink": "Đồ uống",
    "food": "Món ăn",
    "pet": "Thú cưng",
    "interests": "Mối quan tâm",
}

_INTENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("name", re.compile(r"\btên\b|là\s*ai\b", re.IGNORECASE)),
    ("location", re.compile(r"ở\s*đâu|nơi\s*ở|nơi\s*sống|chỗ\s*ở|còn\s*ở|vẫn\s*ở", re.IGNORECASE)),
    ("job", re.compile(r"nghề|nghiệp|công\s*việc|làm\s*gì|làm\s*nghề", re.IGNORECASE)),
    ("style", re.compile(r"style|phong\s*cách|kiểu\s*trả\s*lời|trả\s*lời.*thích|thích.*trả\s*lời", re.IGNORECASE)),
    ("drink", re.compile(r"đồ\s*uống|thức\s*uống|uống\s*gì", re.IGNORECASE)),
    ("food", re.compile(r"món\s*ăn|ăn\s*gì", re.IGNORECASE)),
    ("pet", re.compile(r"nuôi|con\s*gì|corgi|thú\s*cưng|\bchó\b|\bmèo\b", re.IGNORECASE)),
    ("interests", re.compile(r"quan\s*tâm|mối\s*quan\s*tâm|sở\s*thích|thích\s*gì", re.IGNORECASE)),
)

_SUMMARY_REQUEST_RE = re.compile(
    r"tóm\s*tắt|mô\s*tả|biết\s*.*\blà\s*ai\b|\blà\s*ai\b", re.IGNORECASE
)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Student TODO: implement Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        # NOTE: path_for() already appends "profiles", so pass state_dir
        # directly to obtain state/profiles/<user>/User.md (see README).
        self.profile_store = UserProfileStore(self.config.state_dir)
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        self.langchain_agent = None
        if not self.force_offline:
            try:
                self.langchain_agent = self._maybe_build_langchain_agent()
            except Exception:
                self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between deterministic offline mode and the live model path."""

        if not self.force_offline and self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return int(self.thread_tokens.get(thread_id, 0))

    def prompt_token_usage(self, thread_id: str) -> int:
        return int(self.thread_prompt_tokens.get(thread_id, 0))

    def memory_file_size(self, user_id: str) -> int:
        return int(self.profile_store.file_size(user_id))

    def compaction_count(self, thread_id: str) -> int:
        return int(self.compact_memory.compaction_count(thread_id))

    def _read_profile(self, user_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        try:
            text = self.profile_store.read_text(user_id)
        except OSError:
            return facts
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip()
            if key in _PROFILE_LABELS and value:
                facts[key] = value
        return facts

    def _render_profile(self, facts: dict[str, str]) -> str:
        lines = ["# User Profile", ""]
        for key in _PROFILE_KEY_ORDER:
            if facts.get(key):
                lines.append(f"{key}: {facts[key]}")
        return "\n".join(lines) + "\n"

    def _store_updates(self, user_id: str, updates: dict[str, str]) -> dict[str, str]:
        clean = {
            key: value.strip()
            for key, value in updates.items()
            if key in _PROFILE_LABELS
            and isinstance(value, str)
            and value.strip()
        }
        if not clean:
            return self._read_profile(user_id)
        facts = self._read_profile(user_id)
        for key, value in clean.items():
            # Single-value facts (name/location/job/...) are replaced by
            # the newest correction; multi-value preferences accumulate.
            if key in ("style", "interests") and facts.get(key):
                merged = [part for part in facts[key].split(", ") if part]
                for part in [p for p in value.split(", ") if p]:
                    if part not in merged:
                        merged.append(part)
                facts[key] = ", ".join(merged)[:200]
            else:
                facts[key] = value
        self.profile_store.write_text(user_id, self._render_profile(facts))
        return facts

    def _prepare_turn(
        self, user_id: str, thread_id: str, message: str
    ) -> tuple[str, dict[str, str], int]:
        text = "" if message is None else str(message)
        updates = extract_profile_updates(text)
        self._store_updates(user_id, updates)
        self.compact_memory.append(thread_id, "user", text)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        return text, updates, prompt_tokens

    def _finalize_turn(
        self, thread_id: str, response: str, prompt_tokens: int
    ) -> dict[str, Any]:
        self.compact_memory.append(thread_id, "assistant", response)
        agent_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = (
            self.thread_tokens.get(thread_id, 0) + agent_tokens
        )
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )
        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic path: extract -> persist -> compact -> answer."""

        text, updates, prompt_tokens = self._prepare_turn(
            user_id, thread_id, message
        )
        response = self._offline_response(user_id, thread_id, text, updates)
        return self._finalize_turn(thread_id, response, prompt_tokens)

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Live path with the same memory flow; falls back on model error."""

        text, _updates, prompt_tokens = self._prepare_turn(
            user_id, thread_id, message
        )
        try:
            context = self.compact_memory.context(thread_id)
            profile = self.profile_store.read_text(user_id)
            prompt = (
                f"Hồ sơ người dùng:\n{profile}\n"
                f"Tóm tắt hội thoại:\n{context.get('summary', '')}\n"
                f"Tin nhắn: {text}\nTrả lời ngắn gọn."
            )
            result = self.langchain_agent.invoke(prompt)
            response = getattr(result, "content", result)
            response = response if isinstance(response, str) else str(response)
        except Exception:
            response = self._offline_response(user_id, thread_id, text, {})
        return self._finalize_turn(thread_id, response, prompt_tokens)

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Profile + summary + retained recent messages (no full history)."""

        try:
            profile_text = self.profile_store.read_text(user_id)
        except OSError:
            profile_text = ""
        context = self.compact_memory.context(thread_id)
        total = estimate_tokens(profile_text or "")
        total += estimate_tokens(str(context.get("summary", "") or ""))
        messages = context.get("messages", [])
        if isinstance(messages, list):
            for item in messages:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role", "") or "")
                content = item.get("content", "")
                total += estimate_tokens(
                    f"{role} {'' if content is None else str(content)}"
                )
        return total

    def _detect_intents(self, message: str) -> list[str]:
        return [
            key for key, pattern in _INTENT_PATTERNS if pattern.search(message)
        ]

    def _offline_response(
        self,
        user_id: str,
        thread_id: str,
        message: str,
        updates: dict[str, str] | None = None,
    ) -> str:
        """Answer recall questions from `User.md`; acknowledge otherwise."""

        _ = thread_id
        facts = self._read_profile(user_id)
        intents = self._detect_intents(message)
        if intents:
            lines = [
                f"{_PROFILE_LABELS[key]}: {facts[key]}"
                for key in _PROFILE_KEY_ORDER
                if key in intents and facts.get(key)
            ]
            if lines:
                return "Theo hồ sơ mình đã lưu:\n" + "\n".join(f"- {line}" for line in lines)
            missing = ", ".join(_PROFILE_LABELS[key] for key in intents)
            return f"Mình chưa có thông tin {missing} được lưu."
        if _SUMMARY_REQUEST_RE.search(message):
            return self._summarize_profile(facts)
        if updates:
            kept = ", ".join(
                f"{key}={updates[key]}" for key in _PROFILE_KEY_ORDER if key in updates
            )
            return f"Đã ghi nhớ: {kept}."
        if "?" in message:
            return self._summarize_profile(facts)
        return "Đã nhận tin nhắn. Mình sẽ ghi nhớ những thông tin ổn định vào hồ sơ."

    def _summarize_profile(self, facts: dict[str, str]) -> str:
        if not facts:
            return "Mình chưa có thông tin nào được lưu."
        lines = [
            f"- {_PROFILE_LABELS[key]}: {facts[key]}"
            for key in _PROFILE_KEY_ORDER
            if facts.get(key)
        ]
        return "Tóm tắt hồ sơ mình đã lưu:\n" + "\n".join(lines)

    def _maybe_build_langchain_agent(self):
        """Build the live chat model; None when unavailable (offline-safe)."""

        try:
            return build_chat_model(self.config.model)
        except Exception:
            return None
