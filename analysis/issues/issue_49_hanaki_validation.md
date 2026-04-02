# Issue #49: Merge Hanaki & Ozkes (2023) External Validation into Main

## Summary

Split off the Hanaki & Ozkes (2023) external validation pipeline from the broader external validation work (#44) so these results can be merged into main independently.

## Scope

### Scripts
- `analysis/derived/preprocess_hanaki_ozkes.py` — parse 23 Chat=1 zTree session files into tidy parquet
- `analysis/derived/compute_hanaki_embeddings.py` — embed 8,210 French chat messages via text-embedding-3-small
- `analysis/derived/cache_direction_vectors.py` — cache 5 direction vectors as .npy from our embeddings
- `analysis/derived/project_hanaki_embeddings.py` — project external embeddings onto our directions
- `analysis/analysis/regress_hanaki_projections.R` — panel regressions with fixest: session+period FE, pair-clustered SEs
- `analysis/derived/llm_clients/embedding_client.py` — updated embedding client (rewritten for external validation use)

### Output Tables
- `analysis/output/tables/hanaki_external_validation_inv.tex`
- `analysis/output/tables/hanaki_external_validation_pair.tex`

### Tests
- `analysis/tests/test_hanaki_validation.py`
- `analysis/tests/test_hanaki_embeddings.py`
- `analysis/tests/test_hanaki_projections.py`
- `analysis/tests/test_hanaki_regression.py`
- `analysis/tests/test_direction_vectors.py`

## Key Results

- Promise (+5.55***) and Round Liar (-7.98***) projections transfer cleanly to independent French-language bilateral investment game data
- All regressions include log(word_count) control for text-length confound
- Period 0 (practice round) excluded from analysis

## Parent Issue

#44
