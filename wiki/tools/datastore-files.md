---
title: "Datastore Files Reference"
type: tool
tags: [data, csv, datastore, reference]
summary: "Cheat sheet for files in analysis/datastore/derived/ — what they contain, who produces them, who consumes them"
status: active
last_verified: "2026-04-19"
---

## Summary

`analysis/datastore` is a symlink to `~/Library/CloudStorage/Box-Box/SharedFolder_LPCP`. Raw oTree exports live in `datastore/raw/`; derived datasets live in `datastore/derived/`. This article is the file-by-file reference. CSVs are not committed to git — they're regenerated from raw via the `derived/` scripts.

## Raw Data

| File | What |
|---|---|
| `datastore/raw/<session>_data.csv` | One file per session — oTree's per-app data export with all participant fields |
| `datastore/raw/<session>_chat.csv` | One file per session — chat messages with channel format `<segment>-<group>-<round>` |
| `datastore/Rwork/all.csv` | iMotions AFFDEX facial-emotion data, 9,078 rows, page-level granularity |
| `datastore/annotations/` | iMotions annotation strings linking video timestamps to oTree segments/rounds/pages |

## Derived Data (`datastore/derived/`)

### Behavior

| File | Producer | Rows | What |
|---|---|---|---|
| `promise_classifications.csv` | `derived/classify_promises.py` | 5,944 messages | Per-message GPT-5-mini promise label + per-player-round counts |
| `behavior_classifications.csv` | `derived/classify_behavior.py` | 3,520 player-rounds | `made_promise`, `lied_this_round_{20,5}`, `is_liar_{20,5}`, `is_sucker_{20,5}`, `contribution`, `payoff` |
| `liar_buckets.csv` | `derived/liar_buckets.py` | 160 participants | `lie_count`, `liar_bucket` (never/one_time/moderate/severe) |
| `state_classification.csv` | `derived/classify_states.py` | 3,520 | Group-level cooperative state |
| `player_state_classification.csv` | `derived/classify_player_states.py` | 3,520 | Player-level cooperative state (others' contributions only) |

### Contributions & Sentiment

| File | Producer | What |
|---|---|---|
| `contributions.csv` | `derived/build_contributions_xlsx.py` | Player-round contributions + group mates' contributions (added in commit 5b86569) |
| `contributions.xlsx` | same | Excel mirror |
| `sentiment_scores.csv` | `derived/compute_sentiment.py` | 2,298 player-rounds (rounds 2+ only) — VADER pos/neg/neu/compound + `message_count` |
| `participant_payoffs.csv` | `derived/participant_payoffs.py` | One row per participant — final payoffs |

### Merged Panels

| File | Producer | Rows | What |
|---|---|---|---|
| `merged_panel.csv` | `derived/merge_panel_data.py` | 10,683 × 34 | Page-level wide panel: state + sentiment + AFFDEX |
| `dynamic_regression_panel.csv` | `derived/build_dynamic_regression_panel.py` | per-treatment | Pre-built input for `dynamic_regression.R` (issue #57) |
| `issue_20_did_panel.csv` | `derived/issue_20_build_did_panel.py` | event-study panel | Sucker event-time, treated/control flags |
| `issue_17_regression_data.csv` | `derived/merge_regression_data.py` | regression-ready | Sentiment + liar joined |

### Embeddings

| File | What |
|---|---|
| `embeddings_player_round_small.parquet`, `_large.parquet` | OpenAI `text-embedding-3-small` of our chat at player-round level |
| `embeddings_small.parquet`, `_large.parquet` | Same at message level |
| `direction_vectors/*.npy` | Cached centroids: cooperative, promise, homogeneity, round_liar, cumulative_liar |
| `embedding_projections*.csv`, `promise_*`, `homogeneity_*`, `round_liar_*`, `cumulative_liar_*` | Scalar projections onto centroids |
| `hanaki_ozkes_chat_decisions.parquet` | Tidied Hanaki & Ozkes (2023) raw |
| `hanaki_ozkes_embeddings.parquet` | Embedded Hanaki chat (8,210 French messages) |
| `hanaki_ozkes_projections.csv` | Hanaki messages projected onto OUR direction vectors |

## Output Files (`analysis/output/`)

- `output/tables/` — All `.tex` regression tables and summary tables. Filenames usually carry the issue number.
- `output/plots/` — All `.png` and `.pdf` plots. Filenames carry issue number or descriptive name.
- `output/summary_statistics/` — Standalone descriptive stat outputs.

The paper's GitHub Action copies referenced `output/tables/` and `output/plots/` files into `analysis/paper/tables/` and `analysis/paper/plots/` for Overleaf. **Don't manually edit those copies.**

## Conventions

- Keys across most CSVs: `(session_code, segment, round, label)` for player-round; add `page_type` for the merged panel; add `treatment` and `group` where useful.
- `treatment ∈ {1, 2}`. T1 = No Feedback, T2 = Feedback.
- `segment ∈ {supergame1, supergame2, supergame3, supergame4, supergame5}`.
- `label` is letter A–R (no I/O).
- `SESSION_CODE_REMAP` (defined in several scripts) handles a special case for session 03 where the raw CSV's session_code differs from the canonical one.

## Related

- [Analysis Pipeline](analysis-pipeline.md)
- [Merged Panel Construction](../methods/merged-panel.md)
- [Behavior Classification](../methods/behavior-classification.md)
