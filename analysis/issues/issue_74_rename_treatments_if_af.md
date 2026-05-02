# Issue #74: Rename T1/T2 → IF/AF (Individual Feedback / Aggregate Feedback)

## Summary

Rename all treatment labels from `T1`/`Treatment 1`/`Treatment 2` to `IF` (Individual Feedback) and `AF` (Aggregate Feedback) across the paper, analysis scripts, and generated outputs. This makes the treatment names self-documenting and eliminates the need for readers to cross-reference a legend.

## Motivation

- `T1`/`T2` are opaque: readers must look up which is Individual vs Aggregate Feedback.
- `IF`/`AF` are self-describing, matching the mechanism names used in the paper's theory section.
- The original prose at Paper.tex L72 had a labeling inconsistency (IF/AF labels were swapped relative to the mechanism description); this rename fixes that inconsistency.

## Changes

### Paper (Paper.tex)
- Replaced all `T1`/`T2`, `Treatment 1`/`Treatment 2` occurrences with `IF`/`AF`
- Fixed L72 prose: aligned labels with mechanism (IF = individual visibility, AF = aggregate-only)
- Updated equation subscripts: `\text{IF}_i`, `\text{AF}_i`, `\beta_k^{\text{IF}}`, `\beta_k^{\text{AF}}`
- Updated `\includegraphics` references for renamed sentiment distribution plots

### R Scripts (12 files)
- `analysis/analysis/dynamic_regression.R` — relevel factor to "IF"/"AF"; removed dead code (FAMILY_LABELS, order formulas, min/med/max columns)
- `analysis/analysis/issue_12_cdf_plot.R` — updated legend labels
- `analysis/analysis/issue_12_mean_by_round.R` — updated treatment labels
- `analysis/analysis/issue_12_mean_by_segment.R` — updated treatment labels
- `analysis/analysis/issue_12_median_by_round.R` — updated treatment labels
- `analysis/analysis/issue_12_median_by_segment.R` — updated treatment labels
- `analysis/analysis/issue_17_sentiment_liar_regression.R` — updated treatment labels
- `analysis/analysis/issue_59_het_did_coefplot.R` — updated coefplot labels
- `analysis/analysis/issue_59_het_did_regression.R` — updated treatment labels
- `analysis/analysis/issue_67_mean_by_period.R` — updated treatment labels
- `analysis/analysis/issue_72_lying_contagion_regression.R` — relevel factor so coefficients display as "Treatment = IF" instead of "Treatment = 1"
- `analysis/analysis/sentiment_distribution_plot.R` — renamed output files to `_if.png`/`_af.png`

### Python Scripts (9 files)
- `analysis/analysis/issue_64_liar_diff_in_means.py` — updated column headers
- `analysis/analysis/multi_session_analysis.py` — added TODO for matplotlib code (not paper-bound)
- `analysis/analysis/summary_statistics/ss_behavior.py` — updated treatment labels
- `analysis/analysis/summary_statistics/ss_chat.py` — updated treatment labels
- `analysis/analysis/summary_statistics/ss_contributions.py` — updated treatment labels
- `analysis/analysis/summary_statistics/ss_demographics.py` — updated treatment labels
- `analysis/analysis/summary_statistics/ss_experiment_totals.py` — updated treatment labels
- `analysis/analysis/summary_statistics/ss_payoffs.py` — updated treatment labels
- `analysis/analysis/summary_statistics/ss_sentiment.py` — updated treatment labels

### Output Artifacts (~25 files)
- Regenerated all `.tex` summary statistics tables with IF/AF labels
- Regenerated all output tables (`dynamic_regression*.tex`, `issue_72_lying_contagion.tex`, etc.)
- Renamed `sentiment_distribution_t1.png`/`_t2.png` → `sentiment_distribution_if.png`/`_af.png`
- Regenerated all coefplot PDFs with updated labels

### Dynamic Panel Tables (follow-up commit)
- Dropped min/median/max columns from baseline and extended dynamic panel tables
- Baseline: 4 columns → 2 columns (mean only)
- Extended: 12 columns → 6 columns (mean only)
- Removed dead code: `FAMILY_LABELS`, `order_base_rhs`, order-family formulas, unused Wald GoF rows
- Stata DP1 parity preserved: all 6 mean reference rows within 0.006 of Stata output

### Wiki (13 articles)
- Updated all articles referencing T1/T2 to use IF/AF
- `last_verified: 2026-05-01` bumped on all touched articles
- `_index.md` regenerated

## Safety Notes

- Lowercase fixest coefficient regex anchors (`grepl("t1", ...)`) left unchanged — these match CSV column names, not display labels
- CSV filename literals (`*_t1_data.csv`) left unchanged — these are data file names on disk
- `instructions_final.tex` left unchanged (out of scope)

## Validation

- managing-agent review: APPROVED (iteration 2, after fixing 3 summary-stats scripts)
- task-validator: VALIDATED — 0 numerical drift, Paper.pdf builds clean (43 pages, 1.08 MB), 49 IF/AF hits in PDF text, 0 T1/T2 hits
- Stata DP1 parity test passes (all 6 mean reference rows within 0.006)

## Pre-existing Issues (Not Introduced Here)

- Paper.tex L185: pre-existing FOSD logical contradiction (not introduced by this PR)
- Missing citation `isaac1988communication` in `reference.bib` (pre-existing)
