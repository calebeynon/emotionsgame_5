# Issue #19: Swap Strict and Lenient Threshold Definitions

**Date:** 2026-02-01
**Status:** Completed

## Motivation

The original "strict" and "lenient" threshold terminology was semantically inverted and confusing:
- "Strict" (threshold 20) was actually more lenient - easier to classify as a liar
- "Lenient" (threshold 5) was actually stricter - harder to classify as a liar

This issue renames the thresholds to use explicit numeric suffixes that directly reference the threshold values, eliminating ambiguity:
- `_20` suffix: contribution < 20 qualifies as liar/sucker (high threshold, more players qualify)
- `_5` suffix: contribution < 5 qualifies as liar/sucker (low threshold, fewer players qualify)

## Changes Made

### Constants Renamed
- `STRICT_THRESHOLD` → `THRESHOLD_20` (value: 20)
- `LENIENT_THRESHOLD` → `THRESHOLD_5` (value: 5)

### Column Names
- `is_liar_strict` → `is_liar_20`
- `is_liar_lenient` → `is_liar_5`
- `is_sucker_strict` → `is_sucker_20`
- `is_sucker_lenient` → `is_sucker_5`
- `lied_this_period_strict` → `lied_this_period_20`
- `lied_this_period_lenient` → `lied_this_period_5`

### Display Labels
- R scripts: "Is Sucker (Strict)" → "Is Sucker (<20)", "Is Sucker (Lenient)" → "Is Sucker (<5)"
- Paper prose: "strict definition" → "high-threshold definition (< 20)", "lenient definition" → "low-threshold definition (< 5)"

## Files Modified

**Python Source (4):**
- `analysis/derived/behavior_helpers_df.py` - Core constants and functions
- `analysis/derived/behavior_helpers.py` - Re-exports
- `analysis/derived/classify_behavior.py` - Output column names
- `analysis/derived/merge_regression_data.py` - Derived columns

**Tests (3):**
- `analysis/tests/test_behavior_classification.py`
- `analysis/tests/test_behavior_integration.py`
- `analysis/tests/test_merge_regression_data.py`

**R Scripts (4):**
- `analysis/analysis/contribution_regression.R`
- `analysis/analysis/issue_12_table_behavior.R`
- `analysis/analysis/issue_12_table_behavior_aggregate.R`
- `analysis/analysis/issue_17_sentiment_liar_regression.R`

**Documentation (3):**
- `analysis/issues/issue_6_behavior_classification.md`
- `analysis/issues/issue_11_contribution_regression.md`
- `analysis/issues/issue_17_sentiment_liar_regression.md`

**Paper (1):**
- `analysis/paper/preliminary_results_2026_01_28.tex`

**Output Tables (4):**
- `analysis/output/tables/behavior_summary.tex`
- `analysis/output/tables/behavior_summary_aggregate.tex`
- `analysis/output/tables/contribution_regression.tex`
- `analysis/output/tables/issue_17_sentiment_liar_regression.tex`

## Testing

All 244 tests pass after the renaming.

## Impact

This change improves code clarity and reduces ambiguity in the analysis. The numeric suffixes make it immediately clear which threshold value is being applied without needing to remember which term ("strict" or "lenient") corresponds to which number.
