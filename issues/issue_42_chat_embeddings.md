# Issue #42: Chat Embeddings Analysis

## Summary
Build a chat embedding pipeline to analyze cooperative vs. non-cooperative communication patterns in the public goods game, using OpenAI text embeddings for dimensionality reduction, visualization, and regression analysis.

## Background
Existing analysis uses VADER sentiment scores to characterize chat messages, but sentiment captures only positive/negative valence. Embedding-based analysis captures richer semantic structure -- enabling us to study how the *content* of communication (promises, coordination language, deceptive framing) relates to contribution behavior, liar rates, and group cooperation.

## Approach

### Embedding Pipeline
- `embedding_client.py`: OpenAI embedding API client with batching and caching
- `compute_embeddings.py`: Generates embeddings for all chat messages, merges with panel data

### Visualization
- `embedding_plots.py`: UMAP and t-SNE projections colored by cooperative state, group, and contribution level
- `promise_embedding_plots.py`: Same projections split by promise vs. no-promise messages

### Regression Analysis
Six regression scripts predict player contributions from embedding features (UMAP/t-SNE projections), each slicing the data differently:
- `embedding_regression.py`: Base regression across all players
- `promise_embedding_regression.py`: Promise vs. no-promise subgroups
- `homogeneity_embedding_regression.py`: Coordinated vs. independent groups
- `round_liar_embedding_regression.py`: Round-level liar classification
- `cumulative_liar_embedding_regression.py`: Cumulative liar classification
- `group_contribution_embedding_regression.R`: Group-level R regression with fixest

### Analysis Drivers
- `analyze_embeddings.py`, `analyze_promise_embeddings.py`, `analyze_homogeneity_embeddings.py`, `analyze_round_liar_embeddings.py`, `analyze_cumulative_liar_embeddings.py`: End-to-end scripts that compute embeddings, generate plots, and run regressions

### Paper
- `embedding_analysis.tex`: New paper section documenting methodology and results

## Files Changed
- `analysis/derived/llm_clients/embedding_client.py` (new)
- `analysis/derived/llm_clients/__init__.py` (modified)
- `analysis/derived/compute_embeddings.py` (new)
- `analysis/derived/analyze_embeddings.py` (new)
- `analysis/derived/analyze_promise_embeddings.py` (new)
- `analysis/derived/analyze_homogeneity_embeddings.py` (new)
- `analysis/derived/analyze_round_liar_embeddings.py` (new)
- `analysis/derived/analyze_cumulative_liar_embeddings.py` (new)
- `analysis/derived/merge_panel_data.py` (modified -- adds embedding columns)
- `analysis/analysis/embedding_plots.py` (new)
- `analysis/analysis/embedding_regression.py` (new)
- `analysis/analysis/promise_embedding_plots.py` (new)
- `analysis/analysis/promise_embedding_regression.py` (new)
- `analysis/analysis/homogeneity_embedding_regression.py` (new)
- `analysis/analysis/round_liar_embedding_regression.py` (new)
- `analysis/analysis/cumulative_liar_embedding_regression.py` (new)
- `analysis/analysis/group_contribution_embedding_regression.R` (new)
- `analysis/paper/embedding_analysis.tex` (new)
- `analysis/tests/` (20 new test files, ~6000 lines total)
- `pyproject.toml`, `uv.lock` (new dependencies)

## Expected Output
- `analysis/output/plots/`: UMAP and t-SNE embedding visualizations (small/large variants)
- `analysis/output/tables/`: LaTeX regression comparison tables
