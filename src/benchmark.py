from __future__ import annotations

import dataclasses
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read benchmark JSON conversations from disk (UTF-8, read-only)."""

    conv_path = Path(path)
    if not conv_path.is_file():
        raise FileNotFoundError(f"Benchmark data not found: {conv_path}")
    try:
        data = json.loads(conv_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid benchmark JSON: {conv_path}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"Benchmark JSON must be a list: {conv_path}")
    return data


def recall_points(answer: str, expected: list[str]) -> float:
    """Fraction of expected facts present in the answer (case-insensitive)."""

    if not expected:
        return 1.0
    text = "" if answer is None else str(answer).casefold()
    hits = sum(1 for item in expected if str(item).casefold() in text)
    return hits / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight deterministic quality signal, separate from recall."""

    text = "" if answer is None else str(answer)
    if not text.strip():
        return 0.0
    score = 0.4
    if len(text.strip()) >= 20:
        score += 0.2
    if "\n" in text or "-" in text or "•" in text or "1." in text:
        score += 0.2
    if expected and recall_points(text, expected) > 0:
        score += 0.2
    return min(1.0, score)


def _memory_size(agent, user_id: str) -> int:
    func = getattr(agent, "memory_file_size", None)
    if func is None:
        return 0
    try:
        return int(func(user_id))
    except Exception:
        return 0


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent: feed turns per thread, recall in fresh threads."""

    _ = config
    agent_tokens = 0
    prompt_tokens = 0
    compactions = 0
    recall_scores: list[float] = []
    qualities: list[float] = []
    mem_before: dict[str, int] = {}
    users: list[str] = []

    for conv in conversations:
        user_id = str(conv.get("user_id", "user"))
        if user_id not in mem_before:
            mem_before[user_id] = _memory_size(agent, user_id)
            users.append(user_id)
        thread_id = str(conv.get("id", f"conv-{len(users)}"))
        for turn in conv.get("turns", []):
            agent.reply(user_id, thread_id, "" if turn is None else str(turn))
        threads = [thread_id]
        for index, item in enumerate(conv.get("recall_questions", [])):
            recall_thread = f"{thread_id}#recall-{index}"
            threads.append(recall_thread)
            question = item.get("question", "")
            expected = item.get("expected_contains", [])
            answer = agent.reply(
                user_id, recall_thread, "" if question is None else str(question)
            ).get("response", "")
            recall_scores.append(recall_points(answer, expected))
            qualities.append(heuristic_quality(answer, expected))
        for thread in threads:
            agent_tokens += int(agent.token_usage(thread))
            prompt_tokens += int(agent.prompt_token_usage(thread))
            compactions += int(agent.compaction_count(thread))

    growth = sum(_memory_size(agent, user) - mem_before[user] for user in users)
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        response_quality=sum(qualities) / len(qualities) if qualities else 0.0,
        memory_growth_bytes=max(0, growth),
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Render benchmark rows as a table with the six required metrics."""

    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table = [
        [
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            f"{row.recall_score:.3f}",
            f"{row.response_quality:.3f}",
            row.memory_growth_bytes,
            row.compactions,
        ]
        for row in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(table, headers=headers, tablefmt="github")
    except Exception:
        widths = [len(h) for h in headers]
        for row in table:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(str(cell)))
        lines = [
            " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
            "-+-".join("-" * w for w in widths),
        ]
        for row in table:
            lines.append(
                " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))
            )
        return "\n".join(lines)


def _isolated_config(config, workdir: Path, name: str):
    state_dir = workdir / name
    state_dir.mkdir(parents=True, exist_ok=True)
    return dataclasses.replace(config, state_dir=state_dir)


def main() -> None:
    """Run Standard + Long-Context Stress benchmarks, both agents offline."""

    root = Path(__file__).resolve().parent.parent
    config = load_config(root)
    standard = load_conversations(root / "data" / "conversations.json")
    stress = load_conversations(root / "data" / "advanced_long_context.json")
    workdir = Path(tempfile.mkdtemp(prefix="day17-bench-"))

    print("Standard Benchmark")
    print(
        format_rows(
            [
                run_agent_benchmark(
                    "Baseline",
                    BaselineAgent(
                        _isolated_config(config, workdir, "standard-baseline"),
                        force_offline=True,
                    ),
                    standard,
                    config,
                ),
                run_agent_benchmark(
                    "Advanced",
                    AdvancedAgent(
                        _isolated_config(config, workdir, "standard-advanced"),
                        force_offline=True,
                    ),
                    standard,
                    config,
                ),
            ]
        )
    )
    print()
    print("Long-Context Stress Benchmark")
    print(
        format_rows(
            [
                run_agent_benchmark(
                    "Baseline",
                    BaselineAgent(
                        _isolated_config(config, workdir, "stress-baseline"),
                        force_offline=True,
                    ),
                    stress,
                    config,
                ),
                run_agent_benchmark(
                    "Advanced",
                    AdvancedAgent(
                        _isolated_config(config, workdir, "stress-advanced"),
                        force_offline=True,
                    ),
                    stress,
                    config,
                ),
            ]
        )
    )


if __name__ == "__main__":
    main()
