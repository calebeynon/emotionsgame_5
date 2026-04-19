---
title: "Analysis Pipeline"
type: tool
tags: [analysis, data, pipeline, python, R, embeddings, classification]
summary: "Data processing pipeline: experiment_data.py loader, derived classifications, R regressions, and paper output"
status: draft
last_verified: "2026-04-06"
---

## Summary

The analysis module transforms raw oTree CSV exports into research outputs (tables, plots, a LaTeX paper). It has a hierarchical data model (`experiment_data.py`), derived data scripts (embeddings, behavior/promise classification), analysis scripts (Python plots + R regressions), and a video annotation pipeline.

## Key Points

- **Core data model**: `experiment_data.py` provides `Experiment > Session > Segment > Round > Group > Player` hierarchy with built-in VADER sentiment analysis
- **Derived data**: LLM-based classification of promises (OpenAI) and behavior (liar/sucker detection)
- **Statistics**: R scripts using `fixest` for panel regressions with fixed effects
- **Output**: plots, tables, and summary statistics consumed by the LaTeX paper

## Pipeline Stages

```
Raw oTree CSVs (datastore/)
  ↓ experiment_data.py (load + structure)
Experiment object (hierarchical)
  ↓ derived/ scripts
Derived CSVs: promise_classifications, behavior_classifications, embeddings, sentiment
  ↓ analysis/ scripts (Python + R)
Output: plots/, tables/, summary_statistics/
  ↓ paper/ (LaTeX, git subtree → Overleaf)
Published paper
```

## Data Model (`experiment_data.py`)

Hierarchical classes with chat pairing semantics:

- `Experiment` — multi-session container
- `Session` — 16 participants, identified by `session_code`
- `Segment` (supergame) — holds `orphan_chats` (last round's chat with no subsequent contribution)
- `Round` — numbered within segment
- `Group` — 4 players, `group_id_in_subsession`
- `Player` — label A-R, `chat_messages` = chat that influenced THIS contribution
- `ChatMessage` — nickname, body, timestamp, automatic VADER sentiment

**Chat pairing rule**: Chat happens after contribution in oTree, but `experiment_data.py` pairs it with the contribution it influenced (next round). Round 1 has empty `chat_messages`. Last round's chat → `segment.orphan_chats`.

## Derived Data Scripts (`analysis/derived/`)

| Script | Purpose | API |
|--------|---------|-----|
| `classify_promises.py` | LLM promise detection in chat | OpenAI GPT-5-mini |
| `classify_behavior.py` | Liar/sucker classification from promises + contributions | Local logic |
| `compute_embeddings.py` | Chat message embeddings | OpenAI |
| `compute_sentiment.py` | Sentiment scores for chat | VADER (local) |
| `classify_guilt.py` | Guilt emotion classification | LLM |
| `classify_player_states.py` | Player state classification | LLM |
| `merge_regression_data.py` | Merge all derived data into panel datasets | Local |
| `issue_20_build_did_panel.py` | Build difference-in-differences panel | Local |

## Analysis Scripts (`analysis/analysis/`)

### Python Scripts
- `analysis_plots.py` — Main visualization script
- `multi_session_analysis.py` — Cross-session analysis
- `contributions_evolve.py` — Contribution evolution over rounds
- `*_embedding_plots.py` — Embedding visualization (promise, liar, homogeneity)
- `*_embedding_regression.py` — Embedding regression analysis

### R Scripts (issue-organized)
- `contribution_regression*.R` — Core contribution regressions
- `sentiment_contribution_regression.R` — Sentiment impact on contributions
- `issue_12_*.R` — Descriptive statistics, CDF plots, behavior tables
- `issue_17_*.R` — Sentiment-liar regression
- `issue_20_*.R` — Difference-in-differences analysis (liar treatment effects)
- `issue_39_*.R` — Gap tests, negative emotions, decomposition
- `regress_hanaki_projections.R` — Hanaki-Ozkes methodology projections

## Annotation Pipeline (`analysis/annotations/`)

Two-stage process for behavioral video annotation:

1. `build_edited_data_csv.py` — PageTimes CSV + timesheet → normalized event data with timezone-adjusted timestamps
2. `generate_annotations.py` — Event data → annotation markers with duration filtering (>1s)

Participant mapping: IDs 1-16 → labels A1-R1 (skipping I, O).

## Test Suite (`analysis/tests/`)

~60+ test files covering the full pipeline. Run with `uv run pytest` from project root. Key test categories:
- Data model tests (`test_experiment_data_accuracy.py`, `test_contributions.py`, etc.)
- Classification tests (`test_classify_promises.py`, `test_behavior_classification.py`)
- Embedding tests (`test_compute_embeddings.py`, `test_*_embedding_regression.py`)
- Integration tests (`test_integration.py`, `test_panel_integration.py`)

## Related

- [Project Architecture](architecture.md)
- [oTree Experiment Apps](otree-apps.md)
