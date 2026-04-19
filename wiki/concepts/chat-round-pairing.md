---
title: "Chat-Round Pairing Semantics (Critical)"
type: concept
tags: [data-structure, chat, semantics, gotcha, experiment_data]
summary: "Chat from round N is paired with round N+1's contribution because chat happens AFTER contribution in oTree"
status: active
last_verified: "2026-04-19"
---

## Summary

In the oTree experiment, chat happens **after** contribution within a round. But in `experiment_data.py` and all derived analysis, `chat_messages` is paired with the contribution it **influenced** (the next round's). This is the most common source of confusion for new analysis code — get this wrong and you correlate chat with the contribution made before it.

## The Rule

| Where chat occurred (oTree) | Where it lives in `experiment_data.py` | Why |
|---|---|---|
| Round N chat (after Round N contribution) | `Round N+1 → Player → chat_messages` | It influenced Round N+1's contribution |
| Round 1 contribution | `chat_messages = []` (empty) | No prior chat to influence it |
| Last round of segment | `Segment.orphan_chats[label]` | No subsequent contribution to pair with |

## Why It's a Gotcha

The natural assumption ("chat in round N relates to contribution in round N") is wrong. If you build a regression of `contribution ~ sentiment` using a naive merge on `(session, segment, round, label)`, you will measure the wrong causal direction.

## How Derived Pipelines Handle It

- `derived/compute_sentiment.py` produces `sentiment_scores.csv` already in the corrected pairing — sentiment for player `i` in segment `s`, round `r` is the sentiment of chat that **preceded** that round's contribution. As a consequence, round 1 is never present in `sentiment_scores.csv` (2,298 rows, not 3,520).
- `derived/build_dynamic_regression_panel.py` and `derived/merge_panel_data.py` rely on this pairing.
- `analysis/analysis/issue_52_gap_regressions.R` explicitly `shift`s face data forward within `(player, segment)` so the pre-decision face from the previous round's Results page aligns with the current round's contribution.

## How to Verify

Open `experiment_data.py:268-272` — `Segment.orphan_chats` exists *because* of this re-pairing. If you see code merging chat directly on `(round, label)` without a forward shift on chat, suspect a bug.

## Related Issue

Documented in `analysis/issues/issue_7_chat_round_pairing.md` (Issue #7, the original fix).

## Related

- [Analysis Pipeline](../tools/analysis-pipeline.md)
- [Experiment Data Module](../tools/experiment-data-module.md)
