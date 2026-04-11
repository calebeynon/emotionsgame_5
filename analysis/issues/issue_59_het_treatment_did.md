# Issue #59: Heterogeneous Treatment Effects in Sucker DiD Analysis

## Summary

Extends the issue #20 DiD event study to estimate separate treatment effects by treatment group (1 vs 2). The base model pools treatments; this extension interacts treatment with event-study dummies to allow heterogeneous responses to being suckered across treatments.

## Model Specification

**Fully interacted model (single regression):**

```
contribution ~ i(tau, suckered_t1, ref = c(0, 999))
             + i(tau, suckered_t2, ref = c(0, 999))
             | round + segment
```

Where `suckered_t1` and `suckered_t2` are treatment-specific suckered indicators constructed by interacting `got_suckered` with treatment group dummies. Clustered on `cluster_id`.

This produces separate event-study coefficients for Treatment 1 and Treatment 2 in a single model, enabling formal testing of cross-treatment differences.

## Thresholds and Samples

Same as issue #20:
- **Threshold < 20**: groupmate contributed < 20 after promising
- **Threshold < 5**: groupmate contributed < 5 after promising
- **Main sample**: controls are all non-suckered players
- **Robust sample**: controls restricted to always-cooperators

## Input

- `datastore/derived/issue_20_did_panel.csv` (same panel data as issue #20)

## Scripts

| Script | Purpose |
|--------|---------|
| `analysis/issue_59_het_did_regression.R` | Runs 4 regressions (2 thresholds x 2 samples), exports LaTeX table |
| `analysis/issue_59_het_did_coefplot.R` | Generates coefficient plots overlaying Treatment 1 vs Treatment 2 |

## Output

| File | Description |
|------|-------------|
| `output/tables/issue_59_het_did_contribution.tex` | 4-column regression table with T1/T2 coefficients |
| `output/plots/issue_59_het_did_coefplot_20_main.png` | Coefplot, threshold < 20, main sample |
| `output/plots/issue_59_het_did_coefplot_20_robust.png` | Coefplot, threshold < 20, robust sample |
| `output/plots/issue_59_het_did_coefplot_5_main.png` | Coefplot, threshold < 5, main sample |
| `output/plots/issue_59_het_did_coefplot_5_robust.png` | Coefplot, threshold < 5, robust sample |
