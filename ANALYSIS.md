# Day 17 — Memory Systems for AI Agent
## Benchmark Analysis

This analysis compares two memory architectures for the same agent task:
a **Baseline Agent** with thread-local short-term memory only, and an
**Advanced Agent** that adds persistent `User.md` memory plus compact
`summary + recent messages` memory. All numbers below are measured with the
deterministic offline benchmark (`python src/benchmark.py`, `force_offline=True`)
and the offline test suite (`pytest src/test_agents.py -v` → 11 passed).

### Measured results

Standard Benchmark (`data/conversations.json`, 10 conversations):

| Agent | Agent Tokens | Prompt Tokens | Recall | Quality | Memory Growth | Compactions |
|---|---|---:|---:|---:|---:|---:|
| Baseline | 3847 | 25308 | 0.101 | 0.643 | 0 | 0 |
| Advanced | 2060 | 25652 | 1.000 | 1.000 | 458 bytes | 0 |

Long-Context Stress Benchmark (`data/advanced_long_context.json`, 16 long turns):

| Agent | Agent Tokens | Prompt Tokens | Recall | Quality | Memory Growth | Compactions |
|---|---|---:|---:|---:|---:|---:|
| Baseline | 672 | 25159 | 0.000 | 0.600 | 0 | 0 |
| Advanced | 553 | 18169 | 1.000 | 1.000 | 434 bytes | 4 |

Stress prompt-token reduction: 25159 → 18169, about 28%.

## 1. Why does Advanced achieve better recall?

Measured: Standard recall 0.101 → 1.000; Stress recall 0.000 → 1.000.

Mechanism: every user message passes through `extract_profile_updates()`,
stable facts are merged into `User.md`, and `User.md` is keyed by `user_id`
while conversation state is keyed by `thread_id`. A fresh thread therefore
still reads the persistent profile, and `_offline_response()` answers recall
questions from it. The Baseline keys everything by `thread_id`, so a fresh
thread starts with no previous-thread history and cannot recall.

Limitation: Baseline Standard recall is 0.101 rather than exactly 0, but this
is partly a substring-scoring artifact, not true persistent recall — some
recall questions literally mention the answer, and short expected strings such
as `AI` can match inside unrelated words. The gap (0.101 vs 1.000) is still
large enough to demonstrate the architectural difference.

## 2. Why can Advanced cost more in short conversations?

Measured: Standard Prompt Tokens 25308 (Baseline) vs 25652 (Advanced, +1.4%).

Mechanism: Advanced carries the persistent `User.md` profile into every
turn's context, while Standard conversations (~300 tokens per thread) never
reach the compact threshold — compactions are 0 for both agents — so there is
no compaction saving to offset the profile overhead.

Two different costs must be distinguished. Prompt Tokens = context processed
per turn (accumulated). Agent Tokens = response/output tokens only. Compaction
mainly optimizes Prompt Tokens Processed, not Agent Tokens Only. Persistent
memory therefore has overhead: Advanced is not automatically cheaper everywhere.

## 3. Why is compact memory useful for long conversations?

Measured: Stress Prompt Tokens 25159 → 18169 (about 28% lower), with
compactions 0 (Baseline) vs 4 (Advanced) and recall 0.000 vs 1.000.

Mechanism: Baseline reprocesses the full growing conversation history every
turn, so per-turn context grows monotonically. Advanced replaces evicted
older messages with a deterministic `summary` and keeps only recent messages
verbatim, so later turns process `User.md + summary + recent messages`
instead of the entire history.

Trade-off: a summary is lossy compression. An aggressive (low) threshold
compacts early and keeps prompts light but risks discarding detail; a high
threshold preserves raw context longer at higher prompt cost. The default
threshold sits between Standard thread sizes (no useful compaction) and the
Stress thread size (compaction fires), i.e. compact only when context is large
enough for the benefit to exceed the overhead.

## 4. How does persistent memory grow, and what risks appear?

Measured: Standard growth 0 vs 458 bytes; Stress growth 0 vs 434 bytes.

Mechanism: Baseline writes no persistent profile, hence zero growth. Advanced
persists selected long-term facts (name, location, job, style, preferences),
so cross-session recall directly creates storage growth.

Risks: stale facts surviving after a correction, confidently extracted but
incorrect facts, conflicting old-vs-new values, and unbounded profile growth
over many users and sessions. No decay or archival mechanism is implemented
in this lab; `User.md` stays small here only because extraction filters for
stable facts. Production systems would additionally need deduplication, decay,
stale-fact detection, and conflict resolution.

## Bonus — Conflict Handling (90–100 band)

Verified examples: location `Đà Nẵng → Huế`; job `backend engineer → MLOps`.

1. Problem solved. Persistent memory can hold an older fact that conflicts
with a newer user correction (relocation, job change). Keeping both values
would make future recall inconsistent.
2. Benefit. The newest valid correction replaces the stale value for the same
key (single `location:`/`job:` line in `User.md`), so later cross-session
recall returns the current fact. This improves recall consistency; it is one
contributor among several (extraction guards, recall-question protection) and
alone did not cause all of the recall improvement.
3. Risk. If a new message is misclassified as a correction, a correct stored
fact can be overwritten. Stronger memory therefore requires stronger
validation and guardrails (hypothetical/travel/negation/joke/question guards
in the extractor, replace-only-for-single-value-facts in the agent).

## Memory quality vs token cost

The final memory-quality fixes (recall-question protection, union merge of
style/interests) made `User.md` more complete, which helped Advanced reach
recall 1.000. A more complete profile slightly increases persistent prompt
context, yet Stress Prompt Tokens still fell from 25159 to 18169 (about 28%).
This demonstrates the central trade-off: memory quality and context cost pull
in opposite directions, and compact memory is what keeps the total favorable
on long conversations.

## Quality metric limitation

Response quality here is a deterministic offline heuristic (non-empty,
substantive, structured, relevant), not human evaluation and not an LLM judge.
Advantages: offline, reproducible, no API cost. Limitations: it cannot judge
naturalness, nuance, or deep semantic correctness, so it is a supporting
metric only — recall and prompt-cost numbers carry the main conclusions.

## Testing

`pytest src/test_agents.py -v` → 11 passed, all offline with isolated
temporary state and no API keys. Coverage: `User.md` read/write/edit,
compact trigger, cross-session recall, prompt-load reduction, correction
handling, false-positive protection, user isolation, Baseline thread
isolation, Baseline compaction always 0, offline execution, token accounting.

## Conclusion

1. Baseline has no persistent memory → weak cross-session recall (0.101 / 0.000).
2. Advanced adds `User.md` → recall improves to 1.000 on both suites.
3. Full conversation history becomes expensive as context grows (Baseline
Stress prompt load 25159).
4. Compact memory replaces old history with summary + recent messages →
long-context Prompt Tokens decrease (18169, about 28% lower, 4 compactions).
5. Advanced is more capable but also more complex → persistent memory needs
guardrails for stale, incorrect, and conflicting facts.

Short-term memory remembers the current conversation; persistent memory
remembers the user across conversations; compact memory compresses old
conversation history so the agent does not need to carry the entire history
in every prompt.
