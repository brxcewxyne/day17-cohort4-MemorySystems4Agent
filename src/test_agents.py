from __future__ import annotations

import dataclasses
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens

_LONG_PARAGRAPH = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
)


def make_config(tmp_path: Path):
    """Build an isolated config: state under tmp_path, tiny compact threshold."""

    base = load_config(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return dataclasses.replace(
        base,
        state_dir=state_dir,
        compact_threshold_tokens=200,
        compact_keep_messages=4,
    )


def _offline_config(tmp_path: Path):
    """make_config() with API keys stripped for offline-safety tests."""

    config = make_config(tmp_path)
    return dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, api_key=None, base_url=None),
        judge_model=dataclasses.replace(
            config.judge_model, api_key=None, base_url=None
        ),
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""

    store = UserProfileStore(tmp_path / "profiles")
    assert store.file_size("dungct") == 0
    default = store.read_text("dungct")
    assert isinstance(default, str) and default

    path = store.write_text("dungct", "# User Profile\n\nlocation: Đà Nẵng\n")
    assert path.is_file()
    assert store.read_text("dungct") == "# User Profile\n\nlocation: Đà Nẵng\n"
    assert store.file_size("dungct") > 0

    assert store.edit_text("dungct", "Đà Nẵng", "Huế") is True
    assert "Huế" in store.read_text("dungct")
    before = store.read_text("dungct")
    assert store.edit_text("dungct", "Không tồn tại", "X") is False
    assert store.read_text("dungct") == before


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""

    _ = tmp_path
    manager = CompactMemoryManager(threshold_tokens=60, keep_messages=3)
    manager.append("thread-1", "user", "short")
    assert manager.compaction_count("thread-1") == 0
    assert manager.context("thread-1")["summary"] == ""
    for index in range(6):
        manager.append(
            "thread-1", "user", f"long message {index} {_LONG_PARAGRAPH}"
        )
    assert manager.compaction_count("thread-1") > 0
    context = manager.context("thread-1")
    assert context["summary"]
    assert len(context["messages"]) <= 3


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""

    baseline = BaselineAgent(make_config(tmp_path / "base"), force_offline=True)
    advanced = AdvancedAgent(make_config(tmp_path / "adv"), force_offline=True)
    baseline.reply("dungct", "thread-A", "Tôi tên DũngCT.")
    advanced.reply("dungct", "thread-A", "Tôi tên DũngCT.")
    baseline_answer = baseline.reply("dungct", "thread-B", "Tôi tên gì?")["response"]
    advanced_answer = advanced.reply("dungct", "thread-B", "Tôi tên gì?")["response"]
    assert "DũngCT" not in baseline_answer
    assert "DũngCT" in advanced_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""

    baseline = BaselineAgent(make_config(tmp_path / "base"), force_offline=True)
    advanced = AdvancedAgent(make_config(tmp_path / "adv"), force_offline=True)
    messages = [f"turn {index} {_LONG_PARAGRAPH * 3}" for index in range(20)]
    for message in messages:
        baseline.reply("dungct", "thread-L", message)
        advanced.reply("dungct", "thread-L", message)
    assert advanced.compaction_count("thread-L") > 0
    assert baseline.compaction_count("thread-L") == 0
    assert advanced.prompt_token_usage("thread-L") < baseline.prompt_token_usage(
        "thread-L"
    )


def test_correction_handling(tmp_path: Path) -> None:
    """Verify newest correction wins for location and job."""

    advanced = AdvancedAgent(make_config(tmp_path), force_offline=True)
    advanced.reply("dungct", "thread-A", "Mình ở Đà Nẵng.")
    advanced.reply("dungct", "thread-A", "giờ mình đang ở Huế.")
    profile = (tmp_path / "state" / "profiles" / "dungct" / "User.md").read_text(
        encoding="utf-8"
    )
    assert profile.count("location:") == 1
    assert "Huế" in profile
    recall = advanced.reply("dungct", "thread-B", "Hiện tại mình đang ở đâu?")[
        "response"
    ]
    assert "Huế" in recall

    advanced.reply("dungct", "thread-A", "Mình đang làm backend engineer.")
    advanced.reply("dungct", "thread-A", "giờ chuyển sang MLOps engineer.")
    recall_job = advanced.reply("dungct", "thread-C", "Hiện tại mình làm nghề gì?")[
        "response"
    ]
    assert "MLOps" in recall_job


def test_false_positive_memory_protection(tmp_path: Path) -> None:
    """Verify trips and hypotheticals are not persisted."""

    advanced = AdvancedAgent(make_config(tmp_path), force_offline=True)
    advanced.reply("dungct", "thread-A", "Ngày mai tôi đi Hà Nội họp.")
    advanced.reply("dungct", "thread-A", "Nếu tôi làm product manager thì sao?")
    profile = advanced._read_profile("dungct")
    assert profile.get("location") != "Hà Nội"
    assert profile.get("job") != "product manager"


def test_user_isolation(tmp_path: Path) -> None:
    """Verify user A facts do not leak to user B."""

    advanced = AdvancedAgent(make_config(tmp_path), force_offline=True)
    advanced.reply("alice", "thread-A", "Tôi tên DũngCT.")
    answer = advanced.reply("bob", "thread-B", "Tôi tên gì?")["response"]
    assert "DũngCT" not in answer


def test_baseline_thread_isolation(tmp_path: Path) -> None:
    """Verify baseline forgets across threads for the same user."""

    baseline = BaselineAgent(make_config(tmp_path), force_offline=True)
    baseline.reply("dungct", "thread-A", "Tôi tên DũngCT.")
    answer = baseline.reply("dungct", "thread-B", "Tôi tên gì?")["response"]
    assert "DũngCT" not in answer


def test_baseline_compact_always_zero(tmp_path: Path) -> None:
    """Verify baseline never compacts, even on long threads."""

    baseline = BaselineAgent(make_config(tmp_path), force_offline=True)
    for index in range(15):
        baseline.reply("dungct", "thread-L", f"long turn {index} {_LONG_PARAGRAPH}")
    assert baseline.compaction_count("thread-L") == 0
    assert len(baseline.sessions["thread-L"].messages) == 30


def test_offline_safety_no_api_key(tmp_path: Path) -> None:
    """Verify agents construct and reply with no API key in offline mode."""

    config = _offline_config(tmp_path)
    assert config.model.api_key is None
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    assert baseline.langchain_agent is None
    assert advanced.langchain_agent is None
    assert baseline.reply("u", "t", "chào bạn")["response"]
    assert advanced.reply("u", "t", "chào bạn")["response"]


def test_token_counters(tmp_path: Path) -> None:
    """Verify output tokens and growing prompt tokens."""

    baseline = BaselineAgent(make_config(tmp_path / "base"), force_offline=True)
    advanced = AdvancedAgent(make_config(tmp_path / "adv"), force_offline=True)
    for agent in (baseline, advanced):
        first = agent.reply("u", "t", "tin nhắn một")
        assert agent.token_usage("t") > 0
        prompt_before = agent.prompt_token_usage("t")
        assert prompt_before > 0
        agent.reply("u", "t", "tin nhắn hai với nội dung dài hơn khá nhiều")
        assert agent.prompt_token_usage("t") > prompt_before
        assert agent.token_usage("t") >= first["agent_tokens"]
    assert estimate_tokens("hello") > 0
