---
title: "Sentiment Analysis & Sentiment-Contribution Regressions"
type: method
tags: [vader, sentiment, regression, issue-46, issue-17, issue-39]
summary: "VADER sentiment scoring of chat plus the regressions that link it to contributions and to facial emotion"
status: active
last_verified: "2026-04-19"
---

## Summary

VADER (built into `experiment_data.py`) scores each chat message's positive / negative / neutral / compound sentiment. Player-round means feed into the OLS sentiment-contribution regression (issue #46) and several follow-ups examining liar status (issue #17) and the facial-vs-text emotion gap (issue #39).

## Scoring

- Computed in `experiment_data.py` via `nltk.sentiment.SentimentIntensityAnalyzer` (lazy-cached on `ChatMessage.sentiment_scores`).
- `derived/compute_sentiment.py` produces `datastore/derived/sentiment_scores.csv` at the player-round level (2,298 rows; rounds 2+ only because round 1 has no chat under the corrected pairing).
- Aggregations at every hierarchy level: `Player.get_chat_sentiment()`, `Group.get_chat_sentiment()`, `Round.get_chat_sentiment()`, etc.

## Sentiment-Contribution Regression (Issue #46)

```r
contribution ~ sentiment_compound_mean + treatment | round + segment   # baseline
contribution ~ sentiment_compound_mean + message_count + treatment | round + segment   # extended
cluster = ~ session_code + segment + group
```

- Sentiment coefficient ≈ **+2.18\*\*\***, stable across both specs.
- Message count is **not** significant — content drives effect, not volume.
- Treatment effect insignificant in this chatter-only subsample.
- Output: `output/tables/sentiment_contribution_regression.tex`. Used in `Paper.tex` §4.4 (combined with the promise/sucker regression as `contribution_regression_combined.tex`).

## Sentiment × Liar (Issue #17)

`analysis/issue_17_sentiment_liar_regression.R` regresses `sentiment_compound_mean` on liar status (cumulative `is_liar_20`) plus controls. Used to show that liars don't have systematically more negative sentiment in chat — they say cooperative-sounding things and then under-contribute. Pairs naturally with the embedding work and the facial-emotion gap.

## Emotion-Sentiment Gap (Issue #39)

`gap = z(facial_valence) − z(sentiment_compound)` measured on a complete-case panel.

- Sentiment is the stronger predictor of contributions; facial valence alone is not significant.
- The gap differs significantly by **cooperative state** (p = 0.02), **liar status** (p = 0.01), and **sucker status** (p = 0.002).
- A logit of noncooperative behavior on the min-max-normalized gap gives coefficient = 1.84 (p < 0.001).
- Liars show much higher facial valence (9.62) than honest players (3.95) but identical chat sentiment (0.128).

Outputs: 4 dot plots (`emotion_sentiment_gap_by_*.png`), `emotion_sentiment_gap_tests.tex`, 3 regression tables.

### Liar Communication Strategies (sub-result)

GPT-5.4 classified 94 liar chat instances into rhetorical categories:

| Category | % | Notes |
|---|---|---|
| false_promise | 64% | Stated contribution they didn't make |
| deflection_collective | 29% | "We all should..." |
| no_guilt | 23% | No detected strategy |
| manipulation | 13% | Rotation schemes etc. |
| blame_shifting | 12% | Accused others while defecting |
| performative_frustration | 10% | Acted upset while defecting |
| self_justification | 9% | Rationalized own defection |
| duping_delight | 2% | — |
| genuine_guilt | 0% | — |

Genuine guilt is absent in the round-specific liar set; the 3 hand-coded guilt cases all had contribution = 25 (cooperating to make amends) and are filtered out by `lied_this_round_20`.

## Files

- `analysis/sentiment_contribution_regression.R` — main regression.
- `analysis/issue_17_sentiment_liar_regression.R` — liar split.
- `analysis/issue_39_*.R` — gap analysis suite (`common.R`, `plot_dotplots.R`, `gap_tests.R`, `regression_decomposition.R`, `all.R`, `plot_negative_emotions.R`).
- `tests/test_sentiment_contribution_regression.py` — 17 tests.

## Related

- [Behavior Classification](behavior-classification.md)
- [Facial Valence Regressions](facial-valence-regressions.md)
- [Chat-Round Pairing Semantics](../concepts/chat-round-pairing.md)
