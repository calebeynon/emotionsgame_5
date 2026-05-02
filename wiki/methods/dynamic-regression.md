---
title: "Dynamic Panel Regression (Arellano-Bond)"
type: method
tags: [regression, panel, arellano-bond, gmm, dynamic, issue-57, issue-68, issue-74]
summary: "Two-step difference GMM of contribution dynamics. Baseline (2-col, §4.2) + extended (6-col, §4.7) tables. Aligned to Stata Table DP1."
status: active
last_verified: "2026-05-01"
---

## Summary

The headline dynamic specification in the paper. Estimates how a player's change in contribution depends on their own lagged changes and on whether they were above or below their peer-mean contribution previously. Two tables are produced:

- **§4.2 baseline (2 columns)**: one column per treatment (IF, AF), using group-mean deviation regressors only.
- **§4.7 extended (6 columns)**: each baseline column × {Baseline, +Chat, +Chat+Facial}.

Issue #57 introduced chat and facial-emotion extensions. Issue #68 realigned the spec to match the coauthor's Stata `xtabond` Table DP1 (dropped `round2` and `segmentnumber`). Issue #74 dropped the min/median/max peer-deviation columns from both tables and renamed treatment labels T1/T2 → IF/AF.

## Model

The paper writes the specification in levels; the Arellano-Bond two-step estimator differences it internally and uses Windmeijer-corrected robust SEs:

$$
C_{i,t} = \alpha_0 + \sum_{l=1}^{2} \alpha_l\, C_{i,t-l}
+ \beta^{+}\, \text{PosDevInd}_{i,t-1}
+ \beta^{-}\, \text{NegDevInd}_{i,t-1}
+ \gamma\, \mathbf{1}\{\text{Round}_t=1\} + \varepsilon_{i,t}
$$

`PosDevInd` = `contmore_L1` (1 if prior-round contribution exceeded the peer mean), `NegDevInd` = `contless_L1` (1 if it fell short). Instruments: lags 2–5 of $C_{i,t}$. `round1` is the only round dummy; `segmentnumber` and `round2` were dropped in issue #68 to match Stata `xtabond` Table DP1.

### Deviation definition

Only the group-mean reference is used. The panel CSV (`datastore/derived/dynamic_regression_panel.csv`) still contains min/med/max peer stats and their deviation indicators, but they are no longer entered into either regression table after issue #74.

| Variable | Peer reference |
|---|---|
| `contmore_L1` | 1 if prior-round contribution > peer mean (`othercontaverage`) |
| `contless_L1` | 1 if prior-round contribution < peer mean |

## Columns produced

### Baseline table (`output/tables/dynamic_regression_baseline.tex`, 2 cols)

1. IF
2. AF

Both columns match the coauthor's Stata Table DP1 mean-deviation rows within 0.01 (largest diff 0.006 on `round1`).

### Extended table (`output/tables/dynamic_regression_extended.tex`, 6 cols)

Each of the 2 baseline columns appears 3 times: `{Baseline, +Chat, +Chat+Facial}`.

- `+Chat` adds `word_count + made_promise + sentiment_compound_mean`.
- `+Chat+Facial` additionally adds `emotion_valence`, estimated on the emotion-complete subsample (filtered to `!is.na(emotion_valence)`).

## Pipeline

1. **Build panel** (Python): `derived/build_dynamic_regression_panel.py` merges `contributions.csv`, `behavior_classifications.csv`, and `merged_panel.csv` → `datastore/derived/dynamic_regression_panel.csv` (3,520 rows × 59 cols).
   - Adds all min/med/max peer stats and deviation variants (still computed for downstream tests, even though they're no longer in the published tables).
   - Chat NaN (no-message rounds) filled with 0.
   - Emotion NaN left as-is; +facial models use the emotion-complete subsample.
2. **Estimate** (R): `analysis/dynamic_regression.R` reads the pre-built panel, estimates 8 GMM models (2 baseline + 6 extended), exports both LaTeX tables, and prints a Stata Table DP1 parity diagnostic at the end of `main()`.
3. **(Reference) Stata**: Issue-68 reference Stata files are archived under `analysis/issues/issue_68_do1.do` and `analysis/issues/issue_68_table_dp1_reference.{tex,txt}`. Paan's legacy manual Stata output (`analysis/output/tables/dynamic_regression_stata.tex`, 3-column Max/Median/Min + averages) is retained on disk but no longer inputted by `Paper.tex`.

## Output

- `output/tables/dynamic_regression_baseline.tex` — 2 columns (§4.2).
- `output/tables/dynamic_regression_extended.tex` — 6 columns (§4.7).
- N = 1,520 per treatment for all baseline/+Chat columns; +Chat+Facial drops to ≈1,064 (IF) / ≈1,273 (AF) due to AFFDEX availability.
- `dynamic_regression_stata.tex` remains on disk for historical reference but is not `\input` by the paper. The orphaned `dynamic_regression.tex` file was deleted in issue #74.

## Design Notes

- Python handles all merging/derivation; R only estimates and exports. Keeps R scripts pure-statistics.
- Wald tests on coefficient differences use robust `vcovHC()`, not the model vcov (fixed in commit 35b801f).
- `segmentnumber` was dropped in issue #68 after the coauthor's updated Stata spec removed it. The main-paper equation (`eq:dynamic_reg`) was simplified accordingly.
- GoF rows in the rendered table: Observations, AR(1), AR(2), Sargan, and a single `Peer mean pair sum = 0` Wald test (whether the above- and below-peer-mean coefficients sum to zero). The Min/Median/Max pairwise Wald rows were removed in issue #74 along with the columns they tested. Coefficient row labels in the rendered table use intuitive names: `Above peer mean$_{t-1}$` / `Below peer mean$_{t-1}$`.

## Test Coverage

Several panel-construction tests still exercise the min/med/max derived variables (because the panel CSV still contains them), but the regression-output tests that pinned the dropped columns were updated/removed in issue #74. Surviving categories:

- **Panel structure**: `tests/test_dynamic_regression_panel.py`, `tests/test_dynamic_regression_merged_panel.py`, `tests/test_dynamic_regression_minmedmax.py`. These pin row counts, merge integrity, NaN patterns, lag correctness, deviation roundtrips, and min/med/max peer-stat correctness on the **panel CSV**, which still contains all variants.
- **Stata parity (mean rows only after issue #74):** `tests/test_dynamic_regression_parity.py`. Verifies baseline-table coefficients (IF, AF), standard errors, significance stars, and GoF rows against Stata DP1 reference. Per-coefficient tolerance: 0.005 default, Round 1 at 0.01.
- **Pipeline safety**: `tests/test_dynamic_regression_pipeline_safety.py`. Covers `safe_left_merge`, `fill_no_message_rounds`, and `convert_made_promise`.

## Related

- [Merged Panel Construction](merged-panel.md)
- [Main Paper Overview](../papers/main-paper.md)
- [Datastore Files Reference](../tools/datastore-files.md)
