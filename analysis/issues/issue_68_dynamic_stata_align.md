# Issue #68 — Align dynamic panel regression with Stata spec and reorganize paper

## Goal

Align `dynamic_regression.R` with the coauthor's updated Stata `xtabond` spec (Table DP1), add min/med/max deviation variants, emit two LaTeX tables, and reorganize the paper so §4.2 gets a 4-column baseline and a new §4.7 gets a 12-column extended table.

## Reference Stata spec (from `issue_68_do1.do`)

```stata
xtabond contribution contmore_L1 contless_L1 round1 if treatment==1, lags(2) twostep maxldep(4) maxlags(4) vce(robust)
xtabond contribution contmore_L1 contless_L1 round1 if treatment==2, lags(2) twostep maxldep(4) maxlags(4) vce(robust)
xtabond contribution contmoremax_L1 contlessmax_L1 contmoremed_L1 contlessmed_L1 contmoremin_L1 contlessmin_L1 round1 if treatment==1, lags(2) twostep maxldep(4) maxlags(4) vce(robust)
```

Key change vs. prior spec: **drop `round2` and `segmentnumber`**.

## Reference Table DP1 (from `issue_68_table_dp1_reference.txt`)

Three columns:
- (1) T1 min/med/max deviation variant
- (2) T1 mean deviation (baseline)
- (3) T2 mean deviation (baseline)

R coefficients must match within ~0.01 tolerance.

## 4-column §4.2 table layout

Per user decision: extend Table DP1 by adding T2 min/med/max as a 4th column.
1. T1 mean deviations (baseline)
2. T2 mean deviations (baseline)
3. T1 min/med/max deviations
4. T2 min/med/max deviations

## 12-column §4.7 extended table layout

Each of the 4 baseline columns × {Baseline, +Chat, +Chat+Facial} = 12 columns.

## Deliverables

- `build_dynamic_regression_panel.py`: add `othercontmin/max/med`, `contmoremin/max/med`, `contlessmin/max/med`, and `_L1` lags for those six deviation variables
- `dynamic_regression.R`: rewrite to produce 4 baseline + 12 extended models; emit two `.tex` outputs
- `Paper.tex`: update §4.2 equation and `\input{}`; insert new §4.7 "Dynamic Panel with Communication and Emotions" after §4.6 Facial Emotions
- Updated tests covering new panel variables and new regression specs
