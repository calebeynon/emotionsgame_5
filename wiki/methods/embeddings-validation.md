---
title: "Chat Embeddings & Hanaki External Validation"
type: method
tags: [embeddings, openai, umap, projection, external-validation, hanaki, issue-42, issue-49]
summary: "OpenAI text embeddings of chat, centroid projections by behavior class, and external validation on Hanaki & Ozkes (2023) data"
status: active
last_verified: "2026-04-19"
---

## Summary

Issues #42 and #49 build a complete chat-embedding pipeline that goes beyond VADER sentiment by capturing semantic content. Messages are embedded via OpenAI `text-embedding-3-small`, centroids are computed for each behavior class (cooperative / promise / homogeneity / round-liar / cumulative-liar), every message is scalar-projected onto each centroid, and projections are then used as regressors. The same centroids transfer to a French-language bilateral investment experiment (Hanaki & Ozkes 2023).

## Pipeline

| Stage | Script | Output |
|---|---|---|
| Embed our chat | `derived/compute_embeddings.py` | `embeddings_player_round_small.parquet` (and `_large`) |
| Cache centroids | `derived/cache_direction_vectors.py` | `derived/direction_vectors/*.npy` |
| Embed Hanaki chat | `derived/compute_hanaki_embeddings.py` | `hanaki_ozkes_embeddings.parquet` |
| Project our messages | (inside `analyze_*.py`) | `*_embedding_projections*.csv` |
| Project Hanaki messages | `derived/project_hanaki_embeddings.py` | `hanaki_ozkes_projections.csv` |
| Internal regressions | `analysis/group_contribution_embedding_regression.R`, etc. | `output/tables/group_contribution_embedding_regression.tex` |
| External regression | `analysis/regress_hanaki_projections.R` | `output/tables/hanaki_external_validation_inv.tex`, `..._pair.tex` |

## Centroid Definitions

For each binary classification, centroid = mean(embedding | true) − mean(embedding | false), L2-normalized to remove message-length confounds.

- **Cooperative**: others contribute mean ≥ 20.
- **Promise**: a promise was detected in the round.
- **Homogeneity**: all 4 group contributions within 1 ECU of each other.
- **Round-liar**: `lied_this_round_20`.
- **Cumulative-liar**: `is_liar_20`.

Project any new embedding onto a centroid via dot product → a scalar measuring semantic similarity to that class.

## Internal Result

`output/tables/group_contribution_embedding_regression.tex` — group contributions on each projection score. Cooperative and homogeneity are positive and significant. Both liar projections are negative and significant; coefficients > 25 imply the liar effect spreads beyond the lying individual (consistent with the sucker DiD snowball).

## External Result (Hanaki & Ozkes 2023)

Bilateral investment game with French chat (Chat = 1 sessions, 23 sessions, 8,210 messages).

- All projections significant.
- Promise (+5.55\*\*\*) and Round Liar (−7.98\*\*\*) transfer with the predicted sign.
- All regressions include `log(word_count)` as text-length control.
- Period 0 (practice) excluded.

This is the headline external-validity claim of the paper.

## Visualization

- UMAP and t-SNE projections in `output/plots/` (`*_umap_*.png`, `*_tsne_*.png`, `*_projection_dist_*.png`), produced by `analysis/embedding_plots.py` and the per-classification `analysis/{promise,homogeneity,round_liar,cumulative_liar}_embedding_plots.py`.

## API and Cost

- Model: `text-embedding-3-small` (1536-dim).
- Client: `derived/llm_clients/embedding_client.py` with batching + caching.

## Test Coverage

- `tests/test_compute_embeddings.py`
- `tests/test_*_embedding_regression.py` (per classification)
- `tests/test_hanaki_validation.py`, `test_hanaki_embeddings.py`, `test_hanaki_projections.py`, `test_hanaki_regression.py`
- `tests/test_direction_vectors.py`

## Files Touched (selected)

- `derived/`: `compute_embeddings.py`, `compute_hanaki_embeddings.py`, `preprocess_hanaki_ozkes.py`, `cache_direction_vectors.py`, `project_hanaki_embeddings.py`, `analyze_*_embeddings.py`.
- `analysis/`: `embedding_plots.py`, `embedding_regression.py`, `promise_*`, `homogeneity_*`, `round_liar_*`, `cumulative_liar_*`, `group_contribution_embedding_regression.R`, `regress_hanaki_projections.R`.
- `paper/`: `embedding_analysis.tex` (now embedded in main paper section 4.7).

## Related

- [Behavior Classification](behavior-classification.md)
- [Main Paper Overview](../papers/main-paper.md)
