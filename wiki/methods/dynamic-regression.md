---
title: "Dynamic Panel Regression (Arellano-Bond)"
type: method
tags: [regression, panel, arellano-bond, gmm, dynamic, issue-57, issue-68]
summary: "Two-step difference GMM of contribution dynamics. Baseline (4-col, §4.2) + extended (12-col, §4.7) tables. Aligned to Stata Table DP1."
status: active
last_verified: "2026-04-20"
---

## Summary

The headline dynamic specification in the paper. Estimates how a player's change in contribution depends on their own lagged changes and on whether they were above or below their peers' contribution previously. Two tables are produced:

- **§4.2 baseline (4 columns)**: each column is a treatment × deviation-definition combination.
- **§4.7 extended (12 columns)**: each baseline column × {Baseline, +Chat, +Chat+Facial}.

Issue #57 introduced chat and facial-emotion extensions. Issue #68 realigned the spec to match the coauthor's Stata `xtabond` Table DP1 (dropped `round2` and `segmentnumber`, added min/med/max peer-deviation variants).

## Model

Two-step difference GMM (Arellano-Bond) with Windmeijer-corrected robust SEs:

$$
\Delta C_{i,t} = \beta_1 \Delta C_{i,t-1} + \beta_2 \Delta C_{i,t-2}
+ \beta_{pos} \Delta D^+_{i,t-1} + \beta_{neg} \Delta D^-_{i,t-1}
+ \beta_{R1}\, \mathbf{1}\{\text{Round}_t=1\}
+ \varepsilon_{i,t}
$$

Instruments: lags 2–5 of $C_{i,t}$. `round1` is the only round dummy; `segmentnumber` and `round2` were dropped in issue #68 to match Stata `xtabond` Table DP1.

### Deviation definitions

Four flavors of `D^+` / `D^-` are computed; the peer reference point differs:

| Variant | Peer reference | Variables |
|---|---|---|
| Mean | `othercontaverage` (mean of 3 peers) | `contmore_L1`, `contless_L1` |
| Min | `othercontmin` | `contmoremin_L1`, `contlessmin_L1` |
| Med | `othercontmed` (row median of 3 peers) | `contmoremed_L1`, `contlessmed_L1` |
| Max | `othercontmax` | `contmoremax_L1`, `contlessmax_L1` |

The `min/med/max` columns include **all six** `contmore*_L1` and `contless*_L1` terms in one regression; the mean columns include just the two.

## Columns produced

### Baseline table (`output/tables/dynamic_regression_baseline.tex`, 4 cols)

1. T1 mean
2. T2 mean
3. T1 min/med/max
4. T2 min/med/max

T1 min/med/max, T1 mean, and T2 mean match the coauthor's Stata Table DP1 within 0.01 (largest diff 0.006 on `round1`). T2 min/med/max is new (not in Stata reference) — run for symmetry.

### Extended table (`output/tables/dynamic_regression_extended.tex`, 12 cols)

Each of the 4 baseline columns appears 3 times: `{Baseline, +Chat, +Chat+Facial}`.

- `+Chat` adds `word_count + made_promise + sentiment_compound_mean`.
- `+Chat+Facial` additionally adds `emotion_valence`, estimated on the emotion-complete subsample (filtered to `!is.na(emotion_valence)`).

## Pipeline

1. **Build panel** (Python): `derived/build_dynamic_regression_panel.py` merges `contributions.csv`, `behavior_classifications.csv`, and `merged_panel.csv` → `datastore/derived/dynamic_regression_panel.csv` (3,520 rows × 59 cols).
   - Adds all min/med/max peer stats and deviation variants.
   - Chat NaN (no-message rounds) filled with 0.
   - Emotion NaN left as-is; +facial models use the emotion-complete subsample.
2. **Estimate** (R): `analysis/dynamic_regression.R` reads the pre-built panel, estimates 16 GMM models (4 baseline + 12 extended), exports both LaTeX tables, and prints a Stata Table DP1 parity diagnostic at the end of `main()`.
3. **(Reference) Stata**: Issue-68 reference Stata files are archived under `analysis/issues/issue_68_do1.do` and `analysis/issues/issue_68_table_dp1_reference.{tex,txt}`. Paan's legacy manual Stata output (`analysis/output/tables/dynamic_regression_stata.tex`, 3-column Max/Median/Min + averages) is retained on disk but no longer inputted by `Paper.tex`.

## Output

- `output/tables/dynamic_regression_baseline.tex` — 4 columns (§4.2).
- `output/tables/dynamic_regression_extended.tex` — 12 columns (§4.7).
- N = 1,520 per treatment for all baseline/+Chat columns; +Chat+Facial drops to ≈1,064 (T1) / ≈1,273 (T2) due to AFFDEX availability.
- `output/tables/dynamic_regression.tex` and `dynamic_regression_stata.tex` remain on disk for historical reference but are not \input by the paper.

## Design Notes

- Python handles all merging/derivation; R only estimates and exports. Keeps R scripts pure-statistics.
- Wald tests on coefficient differences use robust `vcovHC()`, not the model vcov (fixed in commit 35b801f).
- `segmentnumber` was dropped in issue #68 after the coauthor's updated Stata spec removed it. The main-paper equation (`eq:dynamic_reg`) was simplified accordingly.
- GOF rows in the table: Observations, AR(1), AR(2), Sargan, `Peer mean pair sum = 0` Wald (mean cols), and three pairwise sum-zero Wald tests `Max/Median/Min peer pair sum = 0` (min/med/max cols — each tests whether a single variant's above- and below-peer deviation coefficients sum to zero). The prior `R1+R2=0` Wald test was removed with `round2`. Coefficient row labels in the rendered table use intuitive names: `Above peer mean` / `Below peer mean` for the mean spec and `Above/Below {max, median, min} peer` for the min/med/max spec.

## Test Coverage

Total: **143 tests passing** across three categories.

- **Panel structure (116 tests):** `tests/test_dynamic_regression_panel.py` (46) + `tests/test_dynamic_regression_merged_panel.py` (51) + `tests/test_dynamic_regression_minmedmax.py` (19). Pins row counts, merge integrity, NaN patterns, lag correctness, deviation roundtrips, min/med/max peer-stat correctness, and hand-verified edge rows (tied min=med, mixed more/less, cross-supergame lag).
- **Stata parity (14 tests):** `tests/test_dynamic_regression_significance.py`. Verifies baseline-table coefficients (T1 mean, T2 mean, T1 min/med/max, T2 min/med/max), standard errors, significance stars, and GoF rows (Observations=1520, AR(2)>0.05, Sargan>0.05, pairwise Wald p-values) against Stata DP1 reference. Default tolerance **0.005** with per-coefficient overrides: Round 1 at 0.01 (GMM two-step optimizer noise) and `contmoremax_L1` at 0.006 (Stata 3-decimal rounding boundary). Also pins 21 Chat/Facial coefficients in the extended spec and verifies baseline columns equal their counterparts in the extended table.
- **Pipeline safety (13 tests):** `tests/test_dynamic_regression_pipeline_safety.py`. Covers `safe_left_merge` (duplicate/missing merge-key failures), `fill_no_message_rounds` NaN-bound guard, and `convert_made_promise` NaN error path.

## Related

- [Merged Panel Construction](merged-panel.md)
- [Main Paper Overview](../papers/main-paper.md)
- [Datastore Files Reference](../tools/datastore-files.md)
