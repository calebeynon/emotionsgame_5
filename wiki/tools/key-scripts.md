---
title: "Key Scripts Reference"
type: tool
tags: [scripts, derived, analysis, R, python, reference]
summary: "Per-script index of analysis/derived/ and analysis/analysis/ — what each script does and what it produces"
status: active
last_verified: "2026-05-01"
---

## Summary

Quick reference mapping each major script under `analysis/derived/` (data preparation) and `analysis/analysis/` (estimation + plotting) to its purpose and output. Use this when you know what you want to produce but don't know which file to run.

## `analysis/derived/` — Data Preparation

### Classification

| Script | Output | Purpose |
|---|---|---|
| `classify_promises.py` | `promise_classifications.csv` | GPT-5-mini per-message promise classification |
| `classify_behavior.py` | `behavior_classifications.csv` | Liar / sucker flags from promises + contributions |
| `classify_states.py` | `state_classification.csv` | Group-level cooperative state |
| `classify_player_states.py` | `player_state_classification.csv` | Player-level cooperative state (others only) |
| `classify_guilt.py` | (LLM output) | Guilt emotion classification |
| `liar_buckets.py` | `liar_buckets.csv` | Participant-level liar bucket (never/one/moderate/severe) |

### Embeddings

| Script | Output |
|---|---|
| `compute_embeddings.py` | `embeddings_player_round_small.parquet`, `_large.parquet` |
| `cache_direction_vectors.py` | `direction_vectors/*.npy` |
| `analyze_embeddings.py`, `analyze_promise_embeddings.py`, ... | per-classification projection CSVs |
| `compute_hanaki_embeddings.py` | `hanaki_ozkes_embeddings.parquet` |
| `preprocess_hanaki_ozkes.py` | `hanaki_ozkes_chat_decisions.parquet` |
| `project_hanaki_embeddings.py` | `hanaki_ozkes_projections.csv` |

### Sentiment & Panels

| Script | Output |
|---|---|
| `compute_sentiment.py` | `sentiment_scores.csv` |
| `merge_panel_data.py` | `merged_panel.csv` |
| `merge_regression_data.py` | issue-specific regression-ready CSVs |
| `build_dynamic_regression_panel.py` | `dynamic_regression_panel.csv` |
| `issue_20_build_did_panel.py` | `issue_20_did_panel.csv` |
| `participant_payoffs.py` | `participant_payoffs.csv` |
| `build_contributions_xlsx.py` | `contributions.csv`, `contributions.xlsx` |

### Helpers

| Module | Purpose |
|---|---|
| `behavior_helpers.py` / `behavior_helpers_df.py` | Threshold constants (`THRESHOLD_20`, `THRESHOLD_5`), liar/sucker logic |
| `session_mapping.py` | iMotions session # ↔ oTree session_code, annotation parsing |
| `load_emotion_data.py` | AFFDEX dedup + clean |
| `llm_clients/` | OpenAI, Anthropic, embedding API wrappers |

## `analysis/analysis/` — Estimation & Plotting

### Headline Regressions (used by paper)

| Script | Produces |
|---|---|
| `dynamic_regression.R` | `dynamic_regression.tex` (Arellano-Bond, issue #57) |
| `dynamic_regression.do` | Stata version (kept for cross-check) |
| `contribution_regression_combined.R` | `contribution_regression_combined.tex` (sentiment + promise + sucker) |
| `sentiment_contribution_regression.R` | `sentiment_contribution_regression.tex` (issue #46) |
| `issue_20_did_contribution_regression.R` | Pooled sucker DiD |
| `issue_59_het_did_regression.R` | Heterogeneous sucker DiD (issue #59) |
| `issue_59_het_did_coefplot.R` | `issue_59_het_did_coefplot_*.pdf` |
| `issue_52_gap_regressions.R` | `issue_52_valence_sentiment_gap_regressions.tex` (Table 4) |
| `issue_52_gap_regressions_all_emotions.R` | 13-emotion supplementary tables |
| `issue_64_liar_diff_in_means.py` | `liar_diff_in_means.tex` |
| `regress_hanaki_projections.R` | `hanaki_external_validation_inv.tex`, `..._pair.tex` |
| `group_contribution_embedding_regression.R` | `group_contribution_embedding_regression.tex` |

### Descriptives & Summary Statistics

| Script | Produces |
|---|---|
| `issue_12_mean_by_round.R` / `_by_segment.R` | `mean_contribution_by_*.png` |
| `issue_12_median_by_round.R` / `_by_segment.R` | `median_contribution_by_*.png` |
| `issue_12_cdf_plot.R` | `contribution_cdf_by_treatment.png` |
| `issue_12_table_behavior.R`, `_aggregate.R` | `behavior_summary*.tex` |
| `issue_12_table_contributions.R`, `_aggregate.R` | `contributions_summary*.tex` |
| `sentiment_distribution_plot.R` | `sentiment_distribution_if.png`, `af.png`, `_by_treatment.png` |

### Embedding Plots

| Script | Produces |
|---|---|
| `embedding_plots.py`, `promise_embedding_plots.py`, `homogeneity_embedding_plots.py`, `round_liar_embedding_plots.py`, `cumulative_liar_embedding_plots.py` | UMAP/t-SNE PNG plots in `output/plots/` |

### Issue #39 (emotion-sentiment gap)

`issue_39_all.R` is the master that sources `issue_39_common.R` and runs `issue_39_plot_dotplots.R`, `issue_39_gap_tests.R`, `issue_39_regression_decomposition.R`, `issue_39_plot_negative_emotions.R`.

### Liar Plots & Tables (issue #53)

- `issue_53_distribution_plots.R` → `liar_count_distribution.png`, `sucker_count_distribution.png`
- `issue_53_liar_regression.R` → `liar_conditional_probability.tex` (legacy; superseded by issue #64)
- `issues/issue_53_liar_plots.py` → `sentiment_by_liar_bucket.png`, `emotions_by_liar_bucket.png`, `liar_bucket_summary.tex`

### Validation & Misc

| Script | Purpose |
|---|---|
| `dynamic_regression_validate.R` | Cross-checks the R version against the Stata version |
| `analysis_plots.py`, `multi_session_analysis.py`, `contributions_evolve.py` | Older Python plotting scripts |
| `analysis_up8.py` | Calls Gemini for promise classification (older) |
| `explore_emotion_sentiment.py` | Exploratory only |

## Run Patterns

```bash
# From project root
uv run python analysis/derived/<script>.py
uv run python analysis/analysis/<script>.py
Rscript analysis/analysis/<script>.R
```

Most R scripts assume the working directory is `analysis/` and will look for `datastore/derived/...` relative to that. Most Python scripts use `pathlib.Path` globals at the top.

## Related

- [Analysis Pipeline](analysis-pipeline.md)
- [Datastore Files Reference](datastore-files.md)
- [Project Architecture](architecture.md)
