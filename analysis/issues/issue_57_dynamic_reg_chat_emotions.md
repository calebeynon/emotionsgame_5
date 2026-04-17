# Issue #57: Add Chat Characteristics and Facial Emotions to Dynamic Regression

## Summary
Extend the Arellano-Bond dynamic panel regression to include chat characteristics
(word count, promises, sentiment) and facial emotion (valence) as additional
regressors. Produces a 3x2 table: 3 specification levels (baseline, +chat,
+chat+facial) x 2 treatments.

## Motivation
The existing dynamic regression only captured contribution dynamics and deviation
measures. Chat content and facial expressions carry additional signal about player
intent and emotional state that may predict contribution behavior.

## Changes Made

### New Files
- `analysis/derived/build_dynamic_regression_panel.py`: Merges contributions.csv,
  behavior_classifications.csv, and merged_panel.csv into a single regression-ready
  CSV with chat variables (word_count, made_promise, sentiment_compound_mean) and
  facial emotion (emotion_valence).
- `analysis/tests/test_dynamic_regression_merged_panel.py`: 50 pytest tests
  validating the merged panel (row counts, merge integrity, NaN patterns, lag
  correctness, deviation roundtrips, known values).

### Modified Files
- `analysis/analysis/dynamic_regression.R`: Refactored to read pre-built panel,
  estimate 6 Arellano-Bond GMM models (baseline/+chat/+chat+facial x T1/T2),
  export 6-column LaTeX table with Treatment 1/2 multicolumn headers.
- `analysis/paper/Paper.tex`: Landscape table layout, corrected instrument
  caption (lags 2-5).
- `analysis/paper/tables/dynamic_regression.tex`: Updated table for local
  compilation.

### Key Design Decisions
- Chat variable NaN for no-message rounds (round > 1) filled with 0
- Emotion valence NaN left as-is; +facial models use emotion-complete subsample
- Python handles all data merging/derivation; R only does estimation and export

## Output
- `analysis/output/tables/dynamic_regression.tex`: 6-column LaTeX table
- `analysis/datastore/derived/dynamic_regression_panel.csv`: Merged panel (not committed)

## Issue Link
Closes #57
