---
title: "Merged Panel Construction"
type: method
tags: [data-merging, panel, sentiment, emotion, otree, issue-38]
summary: "How the unified merged_panel.csv (10,683 rows × 34 cols) is built from oTree state, VADER sentiment, and AFFDEX emotion sources"
status: active
last_verified: "2026-04-19"
---

## Summary

`datastore/derived/merged_panel.csv` is the canonical wide panel used by R regressions and many analyses. It joins three sources at page-level granularity (Contribute / Results / ResultsOnly) plus instruction-phase rows. Built by issue #38.

## Three Source Datasets

| Source | File | Rows | Grain |
|---|---|---|---|
| Player state classification | `derived/player_state_classification.csv` | 3,520 | session/segment/round/label |
| VADER sentiment | `derived/sentiment_scores.csv` | 2,298 | session/segment/round/label (rounds 2+ only) |
| iMotions AFFDEX emotions | `datastore/Rwork/all.csv` | 9,078 | session/round/participant/page (raw) |

## Key Challenges

- `all.csv` uses iMotions session numbers (1, 3-11) and composite IDs like `A3` (label + session number). `derived/session_mapping.py` maps these to oTree session codes and treatments and parses annotation strings into segment/round/page_type.
- `all.csv` has duplicate rows per participant per page from multiple iMotions recording segments. `derived/load_emotion_data.py` deduplicates by **averaging non-zero rows**.
- The merged panel **keeps page-level granularity** (Contribute/Results/ResultsOnly), with sentiment and state data joined at round level (so they repeat across the 3 page types).
- An additional 123 instruction-phase rows are appended for the `all_instructions` annotation.

## Build Process

1. `derived/session_mapping.py` — iMotions session # → oTree session code; parse annotation strings.
2. `derived/load_emotion_data.py` — load AFFDEX, dedupe, output clean emotion data keyed by `(session_code, label, segment, round, page_type)`.
3. `derived/merge_panel_data.py` — cross-join state with 3 page types, append instruction rows, LEFT JOIN sentiment + emotion.

## Output Schema

| Column class | Examples |
|---|---|
| Keys | `session_code, treatment, segment, round, group, label, page_type` |
| State | `cooperative, noncooperative, made_promise, ...` |
| Sentiment | `sentiment_compound_mean, message_count, ...` |
| Emotion | `emotion_valence, joy, anger, contempt, sadness, surprise, ...` (13 emotions) |

10,560 game rows (3,520 × 3 page types) + 123 instruction rows = 10,683 total.

## Tests

- `tests/test_merge_panel_data.py` — 745 lines covering annotation parsing, deduplication, session mapping, merge correctness, and edge cases.

## Related

- [Cooperative State Classification](cooperative-state.md)
- [Sentiment Analysis & Sentiment-Contribution Regressions](sentiment-analysis.md)
- [Chat-Round Pairing Semantics](../concepts/chat-round-pairing.md)
