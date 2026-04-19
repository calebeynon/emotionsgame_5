---
title: "Behavior Classification: Promises, Liars, Suckers"
type: method
tags: [classification, liar, sucker, promise, llm, behavior]
summary: "Pipeline that classifies chat as promises (LLM) and players as liars/suckers per round and per segment"
status: active
last_verified: "2026-04-19"
---

## Summary

Two-stage classification pipeline: (1) GPT-5-mini labels each chat message as promise / not-promise with conversational context; (2) Python rules combine promise data with contributions to flag players as liars and suckers under two thresholds. This is the foundation for almost every downstream regression in the project.

## Stage 1: Promise Classification (Issue #2)

- **Script**: `derived/classify_promises.py`
- **Model**: OpenAI GPT-5-mini, ~$0.21 total run cost
- **Context-aware rules**: Proposals only count if accepted; "ok"/"yes" only count if responding to proposals; "same"/"me too" only count if responding to commitments.
- **Output**: `datastore/derived/promise_classifications.csv` — 5,944 messages, 1,281 promises (21.6% promise rate).
- **Validation**: 84.9% agreement with Anthropic in early dual-classifier runs (issue #2); OpenAI chosen for accuracy. Final system uses only OpenAI.

## Stage 2: Liar / Sucker Flags (Issue #6)

- **Script**: `derived/classify_behavior.py` (helpers: `behavior_helpers.py`, `behavior_helpers_df.py`)
- **Output**: `datastore/derived/behavior_classifications.csv` — 3,520 player-rounds (10 sessions × 16 players × 22 rounds).

### Definitions

| Flag | Meaning |
|---|---|
| `made_promise` | Player made a promise this round. |
| `lied_this_round_20` / `_5` | Player promised AND contributed below threshold (20 or 5) in this round. |
| `is_liar_20` / `_5` | Player has lied at least once **earlier in the current segment** (cumulative within segment, resets between segments). |
| `is_sucker_20` / `_5` | Player contributed 25 in a round when a groupmate broke their promise (cumulative within segment, resets between segments). |

### Persistence Rules

- Round 1 of each segment: all flags are `False` (no prior history).
- Once a flag is set, it persists through the rest of the segment.
- Flags reset to `False` at the start of every new segment.
- See `concepts/glossary.md` for the cumulative vs. round-specific distinction.

## Stage 3: Liar Buckets (Issue #53)

- **Script**: `derived/liar_buckets.py`
- **Output**: `datastore/derived/liar_buckets.csv` (160 rows, one per participant).
- **Logic**: Sum `lied_this_round_20` across all 22 rounds per participant. Bucket: `never` (0), `one_time` (1), `moderate` (2-3), `severe` (4+). Used for cross-participant comparisons of sentiment, emotion, and other characteristics.

## Threshold Naming (Issue #19)

The earlier suffixes `_strict` (= threshold 20) and `_lenient` (= threshold 5) were swapped to `_20` and `_5` because "strict" was actually *more* lenient toward calling someone a liar. Always use the numeric suffixes in new code.

## Key Empirical Facts (for AI agents)

- 21.6% of messages are promises.
- 105 of 160 participants never lie across the entire experiment.
- T1 participants are 16.2 pp more likely to ever lie than T2 (issue #64, p = 0.031).
- A player's bucket reflects *total* lying frequency across the experiment, not per-segment.

## Test Coverage

- `tests/test_classify_promises.py`
- `tests/test_behavior_classification.py` (34 cases)
- `tests/test_behavior_integration.py`
- `tests/test_liar_buckets.py`

## Related

- [Project Glossary](../concepts/glossary.md)
- [Liar Diff-in-Means (Issue #64)](liar-diff-in-means.md)
- [Cooperative State Classification](cooperative-state.md)
- [Liar Flag: Cumulative vs Round-Specific](liar-flag-comparison.md)
