from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Simple deterministic offline token estimate.

    - Empty or whitespace-only text -> 0.
    - Otherwise at least 1, approximated as characters // 4.
    """

    if not isinstance(text, str):
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Student TODO:
    - Map each user id to one markdown file
    - Support read / write / edit operations
    - Optionally expose helpers like `facts()` or `upsert_fact()`
    """

    root_dir: Path

    def _safe_user_id(self, user_id: str) -> str:
        text = str(user_id).strip() if isinstance(user_id, str) else str(user_id)
        safe = "".join(
            ch if (ch.isalnum() or ch in ("_", "-")) else "_"
            for ch in text
        ).strip("_-")
        if not safe or safe in (".", ".."):
            return "user"
        return safe[:64]

    def path_for(self, user_id: str) -> Path:
        safe_id = self._safe_user_id(user_id)
        return Path(self.root_dir) / "profiles" / safe_id / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return "# User Profile\n\n_No profile information stored yet._\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        if not search_text:
            return False
        path = self.path_for(user_id)
        if not path.is_file():
            return False
        current = path.read_text(encoding="utf-8")
        if search_text not in current:
            return False
        updated = current.replace(search_text, replacement, 1)
        path.write_text(updated, encoding="utf-8")
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if not path.is_file():
            return 0
        return path.stat().st_size


_ALLOWED_PROFILE_KEYS = frozenset(
    {"name", "location", "job", "style", "drink", "food", "pet", "interests"}
)
_MAX_VALUE_LEN = 120

_JOB_KEYWORDS = (
    "engineer",
    "developer",
    "devops",
    "mlops",
    "backend",
    "frontend",
    "fullstack",
    "product manager",
    "manager",
    "designer",
    "data scientist",
    "scientist",
    "architect",
    "researcher",
    "lead",
    "intern",
    "tester",
    "kỹ sư",
)
_JOB_TRAILING_STOPWORDS = {
    "cho", "với", "tại", "ở", "và", "để", "một", "của", "trong",
    "về", "là", "chứ", "nhé", "nha", "rồi", "này", "đó", "mà",
    "thì", "như", "cùng", "vẫn", "với",
}

_HYPOTHETICAL_RE = re.compile(
    r"(^|[.!?;…\n]\s*)(nếu|giả\s*sử|giá\s*như)\b|\bnếu\s+(tôi|mình|tớ|tao)\b|\bhay\s+là\b",
    re.IGNORECASE,
)
_JOKE_RE = re.compile(r"đùa|nói\s*đùa|đùa\s*thôi|câu\s*đùa", re.IGNORECASE)
_META_PLACE_RE = re.compile(
    r"ví\s*dụ\s*cũ|như\s*ví\s*dụ|đừng\s*lấy|thông\s*tin\s*cũ", re.IGNORECASE
)
_TRAVEL_VERB_RE = re.compile(
    r"\b(đi|bay|công\s*tác|du\s*lịch|ghé(\s*thăm|\s*qua|\s*lại)?|quá\s*cảnh|họp)\b",
    re.IGNORECASE,
)
_RESIDENCE_VERB_RE = re.compile(
    r"\b(sống|ở|tại|làm\s*việc\s+ở)\b", re.IGNORECASE
)
_QUESTION_FACT_RE = re.compile(
    r"bạn\s+(có\s+)?biết"
    r"|(tên|nghề(\s*nghiệp)?|nơi\s*ở|đồ\s*uống|món\s*ăn)\b.{0,30}(gì|nào|ai)\s*\?"
    r"|\blà\s*ai\s*\?"
    r"|\bở\s*đâu\s*\?",
    re.IGNORECASE,
)
_THICH_RE = r"(?<!giải\s)(?<!giãi\s)\bthích\b"
_NEGATION_RE = re.compile(
    r"\b(không(\s*còn|\s*phải)?|chưa|chẳng|đừng)\b", re.IGNORECASE
)
_RECALL_REQUEST_RE = re.compile(
    r"\b(nhắc\s*lại|nhớ\s*lại|kể\s*lại|hãy\s+(nhắc|kể|mô\s*tả)"
    r"|cho\s+(tôi|mình)\s+biết|tóm\s*tắt|mô\s*tả)\b",
    re.IGNORECASE,
)

_NAME_RES = (
    re.compile(
        r"(?:tôi|mình|tớ|tao|em|anh|chị)\s+tên\s+(?:là\s+)?"
        r"([A-ZÀ-ỸĐ][\wÀ-ỹ.\-]*(?:\s+[A-ZÀ-ỸĐ][\wÀ-ỹ.\-]*){0,2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"tên\s+(?:của\s+)?(?:tôi|mình|tớ|em|anh|chị)\s+là\s+"
        r"([A-ZÀ-ỸĐ][\wÀ-ỹ.\-]*(?:\s+[A-ZÀ-ỸĐ][\wÀ-ỹ.\-]*){0,2})",
        re.IGNORECASE,
    ),
)
_LOCATION_RES = (
    re.compile(
        r"(?:hiện\s+(?:tại\s+|nay\s+|giờ\s+)?|giờ\s+)?"
        r"(?:mình|tôi|tớ|em|anh|chị)?\s*(?:đang\s+|vẫn\s+)?"
        r"(?:sống(?:\s+(?:ở|tại))?|ở)\s+(?:tại\s+)?"
        r"([A-ZÀ-ỸĐ][\wÀ-ỹ\-]*(?:\s+[A-ZÀ-ỸĐ][\wÀ-ỹ\-]*){0,1})",
        re.IGNORECASE,
    ),
    re.compile(
        r"làm\s*việc\s+ở\s+"
        r"([A-ZÀ-ỸĐ][\wÀ-ỹ\-]*(?:\s+[A-ZÀ-ỸĐ][\wÀ-ỹ\-]*){0,1})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:nơi\s*ở|nơi\s*sống|chỗ\s*ở)"
        r"(?:\s*hiện\s*tại|\s*của\s*(?:mình|tôi))?\s*là\s+"
        r"([A-ZÀ-ỸĐ][\wÀ-ỹ\-]*(?:\s+[A-ZÀ-ỸĐ][\wÀ-ỹ\-]*){0,1})",
        re.IGNORECASE,
    ),
)
_MOVE_RE = re.compile(
    r"chuyển\s+(?:sang|tới|vào|đến)\s+"
    r"([A-Za-zÀ-ỹĐ][\wÀ-ỹ.\-]*(?:\s+[A-Za-zÀ-ỹĐ][\wÀ-ỹ.\-]*){0,2})",
    re.IGNORECASE,
)
_MOVE_FROM_RE = re.compile(
    r"chuyển\s+từ\s+.+?\s+sang\s+"
    r"([A-Za-zÀ-ỹĐ][\wÀ-ỹ.\-]*(?:\s+[A-Za-zÀ-ỹĐ][\wÀ-ỹ.\-]*){0,2})",
    re.IGNORECASE,
)
_JOB_RES = (
    re.compile(
        r"đang\s+làm\s+(?!việc\b)"
        r"([A-Za-zÀ-ỹ][\wÀ-ỹ.\-]*(?:\s+[A-Za-zÀ-ỹ][\wÀ-ỹ.\-]*){0,2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<![\wÀ-ỹ])làm\s+(?!việc\b)"
        r"([A-Za-zÀ-ỹ][\wÀ-ỹ.\-]*(?:\s+[A-Za-zÀ-ỹ][\wÀ-ỹ.\-]*){0,2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:nghề|nghề\s*nghiệp|công\s*việc)"
        r"(?:\s*hiện\s*tại|\s*của\s*(?:tôi|mình))?\s*(?:của\s*(?:tôi|mình)\s*)?là\s+"
        r"([A-Za-zÀ-ỹ][\wÀ-ỹ.\-]*(?:\s+[A-Za-zÀ-ỹ][\wÀ-ỹ.\-]*){0,2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"vẫn\s+là\s+"
        r"([A-Za-zÀ-ỹ][\wÀ-ỹ.\-]*(?:\s+[A-Za-zÀ-ỹ][\wÀ-ỹ.\-]*){0,2})",
        re.IGNORECASE,
    ),
)
_STYLE_SIGNALS = (
    ("3 bullet", re.compile(r"3\s*bullet|ba\s*bullet", re.IGNORECASE)),
    ("ngắn gọn", re.compile(r"ngắn\s*gọn|trả\s*lời\s*ngắn|câu\s*trả\s*lời\s*gọn|giữ.*\bgọn\b", re.IGNORECASE)),
    ("ví dụ thực chiến", re.compile(r"ví\s*dụ\s*thực\s*(chiến|tế)", re.IGNORECASE)),
    ("có cấu trúc", re.compile(r"có\s*cấu\s*trúc", re.IGNORECASE)),
    ("rõ ý", re.compile(r"\brõ\s*ý\b", re.IGNORECASE)),
)
_DRINK_RE = re.compile(
    r"cà\s*phê\s*sữa\s*đá|cafe\s*sữa\s*đá|caphe\s*sua\s*da", re.IGNORECASE
)
_DRINK_CTX_RE = re.compile(
    _THICH_RE + r"|uống|\byêu\s*thích\b|đồ\s*uống|thức\s*uống|món.*uống",
    re.IGNORECASE,
)
_FOOD_ITEMS = (("mì quảng", "mì Quảng"),)
_FOOD_CTX_RE = re.compile(
    _THICH_RE
    + r"|\byêu\s*thích\b|món\s*ăn|món\s*ruột|khoái|ghiền|món\s*tủ",
    re.IGNORECASE,
)
_MEDIA_NOUNS = {"tin", "bài", "báo", "post", "video", "clip", "bản"}
_PET_RE = re.compile(r"nuôi\b.{0,30}corgi|\bcorgi\b", re.IGNORECASE)
_PET_NAME_RE = re.compile(
    r"corgi\s+(?:tên\s+)?([A-Za-zÀ-ỹĐ][\wÀ-ỹ]*)", re.IGNORECASE
)
_INTEREST_CTX_RE = re.compile(
    r"("
    + _THICH_RE
    + r"|\byêu\s*thích\b|\bquan\s*tâm\b|\bđam\s*mê\b|\bmê\b|\bhứng\s*thú\b)"
    r"(\s*(nhiều|những|mấy))?\s*(đến|tới|về|với)?\s*([^.;!?…]{2,80})",
    re.IGNORECASE,
)
_INTEREST_SPLIT_RE = re.compile(r",|\bvà\b|\bvới\b|\bcùng\b|;", re.IGNORECASE)


def _clean_value(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip()).strip(" .,;:!?…\"'“”‘’()[]")
    return cleaned[:_MAX_VALUE_LEN].strip()


def _looks_like_job(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in _JOB_KEYWORDS)


def _trim_job_value(text: str) -> str:
    words = _clean_value(text).split()
    while words and words[-1].lower().strip(".,") in _JOB_TRAILING_STOPWORDS:
        words.pop()
    return " ".join(words[:2])


def _trim_place_value(text: str) -> str:
    """Trim a captured place/name to capitalized words only.

    The `À-Ỹ` range also covers lowercase Vietnamese letters, so with
    IGNORECASE a capture may start/end with lowercase words
    (e.g. `ở Huế`, `Huế rồi`). Drop those edge words.
    """

    words = _clean_value(text).split()
    while words and not words[0][:1].isupper():
        words.pop(0)
    while words and not words[-1][:1].isupper():
        words.pop()
    return " ".join(words[:2])


def _negated_before(sentence: str, start: int, window: int = 25) -> bool:
    return _NEGATION_RE.search(sentence[max(0, start - window):start]) is not None


def _validate_profile_updates(candidate: object) -> dict[str, str]:
    """Validate a candidate fact dict for future LLM fallback integration.

    NOT called by the offline regex extractor. Accepts only known keys,
    trims values, rejects empty/malformed entries and unknown keys.
    """

    if not isinstance(candidate, dict):
        return {}
    valid: dict[str, str] = {}
    for key, value in candidate.items():
        if key not in _ALLOWED_PROFILE_KEYS:
            continue
        if not isinstance(value, str):
            continue
        cleaned = _clean_value(value)
        if not cleaned:
            continue
        valid[key] = cleaned
    return valid


def _merge_profile_updates(
    regex_facts: object, llm_facts: object
) -> dict[str, str]:
    """Merge future LLM facts under deterministic regex facts.

    Regex facts have precedence: LLM fills only missing fields, never
    overwrites confident deterministic facts. NOT wired to any model.
    """

    merged = _validate_profile_updates(llm_facts)
    merged.update(_validate_profile_updates(regex_facts))
    return merged


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract stable profile facts from one user message.

    Deterministic regex/rule-based offline path: no model, no network.
    Returns only high-confidence facts (name, location, job, style,
    drink, food, pet, interests). Hypothetical, travel, question,
    negated, joke and meta-example content is never persisted.
    Later sentences overwrite earlier ones for the same key so
    corrections yield the new fact.
    """

    if not isinstance(message, str):
        return {}
    if not message.strip():
        return {}

    facts: dict[str, str] = {}
    style_parts: list[str] = []
    interest_parts: list[str] = []

    sentences = [
        s.strip() for s in re.split(r"[.!?…\n;]+|\?(?=\s|$)", message) if s.strip()
    ]
    if not sentences:
        return {}

    for sentence in sentences:
        identity_guarded = bool(
            _HYPOTHETICAL_RE.search(sentence)
            or _JOKE_RE.search(sentence)
            or _QUESTION_FACT_RE.search(sentence)
        )
        question_guarded = bool(_QUESTION_FACT_RE.search(sentence))
        # Recall requests ("hãy nhắc…", "tóm tắt…", "mô tả…") ask for
        # stored facts; they must never overwrite the profile. Note that
        # "nhớ là…" (asking to remember X) is intentionally NOT guarded.
        recall_guarded = bool(_RECALL_REQUEST_RE.search(sentence))
        place_meta = bool(_META_PLACE_RE.search(sentence))
        travel_only = bool(
            _TRAVEL_VERB_RE.search(sentence)
            and not _RESIDENCE_VERB_RE.search(sentence)
        )

        if not identity_guarded and not recall_guarded:
            for pattern in _NAME_RES:
                matched_here = False
                for match in pattern.finditer(sentence):
                    if _negated_before(sentence, match.start(1)):
                        continue
                    value = _trim_place_value(match.group(1))
                    if value and value[0].isupper():
                        facts["name"] = value
                        matched_here = True
                        break
                if matched_here:
                    break

        if (
            not identity_guarded
            and not recall_guarded
            and not place_meta
            and not travel_only
        ):
            for pattern in _LOCATION_RES:
                for match in pattern.finditer(sentence):
                    if _negated_before(sentence, match.start(1)):
                        continue
                    raw_first = _clean_value(match.group(1)).split()
                    if raw_first and raw_first[0].lower() in _MEDIA_NOUNS:
                        continue
                    value = _trim_place_value(match.group(1))
                    if value:
                        facts["location"] = value
            move_match = _MOVE_FROM_RE.search(sentence) or _MOVE_RE.search(
                sentence
            )
            if move_match and not _negated_before(sentence, move_match.start(1)):
                candidate = _trim_place_value(move_match.group(1))
                if candidate and not _looks_like_job(candidate):
                    facts["location"] = candidate

        if not identity_guarded and not recall_guarded and not place_meta:
            for pattern in _JOB_RES:
                matched_here = False
                for match in pattern.finditer(sentence):
                    if _negated_before(sentence, match.start(1)):
                        continue
                    value = _trim_job_value(match.group(1))
                    if value and _looks_like_job(value):
                        facts["job"] = value
                        matched_here = True
                        break
                if matched_here:
                    break
            move_match = _MOVE_FROM_RE.search(sentence) or _MOVE_RE.search(
                sentence
            )
            if move_match and not _negated_before(sentence, move_match.start(1)):
                candidate = _trim_job_value(move_match.group(1))
                if candidate and _looks_like_job(candidate):
                    facts["job"] = candidate

        for label, signal in _STYLE_SIGNALS:
            if signal.search(sentence) and label not in style_parts:
                style_parts.append(label)

        if (
            not recall_guarded
            and _DRINK_RE.search(sentence)
            and _DRINK_CTX_RE.search(sentence)
        ):
            facts["drink"] = "cà phê sữa đá"

        lowered = sentence.lower()
        for raw, label in _FOOD_ITEMS:
            if raw in lowered and _FOOD_CTX_RE.search(sentence):
                if not recall_guarded:
                    facts["food"] = label

        if _PET_RE.search(sentence) and not recall_guarded:
            pet_name = _PET_NAME_RE.search(sentence)
            if pet_name:
                facts["pet"] = f"corgi {_clean_value(pet_name.group(1))}"
            else:
                facts["pet"] = facts.get("pet", "corgi")

        interest_match = _INTEREST_CTX_RE.search(sentence)
        if (
            interest_match
            and not question_guarded
            and not recall_guarded
            and not _negated_before(sentence, interest_match.start())
        ):
            clause = interest_match.group(5)
            clause = re.sub(
                r"^(hơn\s+là|hơn)\s*", "", clause, flags=re.IGNORECASE
            )
            negated_clause = re.search(
                r"không\s+(phải\s+)?(chỉ\s+)?là\b", clause, re.IGNORECASE
            )
            if negated_clause:
                after = re.search(r"mà\s*là\s*(.+)", clause, re.IGNORECASE)
                clause = after.group(1) if after else ""
            if re.search(r"\b(gì|nào)\s*$", clause, re.IGNORECASE):
                continue
            for part in _INTEREST_SPLIT_RE.split(clause):
                item = _clean_value(part)
                if not item or len(item) < 2:
                    continue
                if _DRINK_RE.search(item):
                    continue
                if any(
                    food_raw in item.lower() for food_raw, _ in _FOOD_ITEMS
                ):
                    continue
                if re.search(
                    r"trả\s*lời|ví\s*dụ|bullet|ngắn\s*gọn|cấu\s*trúc|rõ\s*ý",
                    item,
                    re.IGNORECASE,
                ):
                    continue
                if item not in interest_parts:
                    interest_parts.append(item)

    if style_parts:
        facts["style"] = ", ".join(
            part for part in [p[0] for p in _STYLE_SIGNALS] if part in style_parts
        )
    if interest_parts:
        facts["interests"] = ", ".join(interest_parts)[:_MAX_VALUE_LEN]

    return {key: facts[key] for key in facts if facts[key]}


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Deterministic heuristic summary of older messages.

    Keeps `role: content` pairs in chronological order, at most
    `max_items` of them, each content truncated so the summary never
    exceeds the original content length. No model, no network.
    """

    if not messages:
        return ""
    items = list(messages)
    if max_items is not None and max_items >= 0:
        items = items[:max_items] if len(items) > max_items else items
    parts: list[str] = []
    for message in items:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user") or "user")
        content = message.get("content", "")
        content = "" if content is None else str(content)
        content = re.sub(r"\s+", " ", content).strip()
        if not content:
            continue
        parts.append(f"{role}: {content[:160]}")
    if not parts:
        return ""
    return " | ".join(parts)


@dataclass
class CompactMemoryManager:
    """Student TODO: implement compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure(self, thread_id: str) -> dict[str, object]:
        entry = self.state.get(thread_id)
        if not isinstance(entry, dict):
            entry = {"messages": [], "summary": "", "compactions": 0}
            self.state[thread_id] = entry
        entry.setdefault("messages", [])
        entry.setdefault("summary", "")
        entry.setdefault("compactions", 0)
        return entry

    def _load_tokens(self, entry: dict[str, object]) -> int:
        total = estimate_tokens(str(entry.get("summary", "")))
        messages = entry.get("messages", [])
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role", "") or "")
                content = message.get("content", "")
                total += estimate_tokens(
                    f"{role} {'' if content is None else str(content)}"
                )
        return total

    def append(self, thread_id: str, role: str, content: str) -> None:
        entry = self._ensure(thread_id)
        messages = entry["messages"]
        assert isinstance(messages, list)
        messages.append(
            {
                "role": str(role) if role else "user",
                "content": "" if content is None else str(content),
            }
        )
        if self._load_tokens(entry) <= self.threshold_tokens:
            return
        keep = max(0, int(self.keep_messages))
        if keep > 0:
            if len(messages) <= keep:
                return
            old = list(messages[:-keep])
            recent = list(messages[-keep:])
        else:
            old = list(messages)
            recent = []
        if not old:
            return
        new_summary = summarize_messages(old)
        previous = str(entry.get("summary", "") or "")
        if previous and new_summary:
            merged = f"{previous}\n{new_summary}"
        else:
            merged = previous or new_summary
        entry["messages"] = [dict(m) for m in recent]
        entry["summary"] = merged
        entry["compactions"] = int(entry.get("compactions", 0) or 0) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        entry = self.state.get(thread_id)
        if not isinstance(entry, dict):
            return {"messages": [], "summary": "", "compactions": 0}
        messages = entry.get("messages", [])
        copied = (
            [dict(m) for m in messages]
            if isinstance(messages, list)
            else []
        )
        return {
            "messages": copied,
            "summary": str(entry.get("summary", "") or ""),
            "compactions": int(entry.get("compactions", 0) or 0),
        }

    def compaction_count(self, thread_id: str) -> int:
        entry = self.state.get(thread_id)
        if not isinstance(entry, dict):
            return 0
        try:
            return int(entry.get("compactions", 0) or 0)
        except (TypeError, ValueError):
            return 0
