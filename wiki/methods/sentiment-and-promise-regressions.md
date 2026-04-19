---
title: "Promise, Sucker, and Treatment Effects on Contributions"
type: method
tags: [regression, ols, promise, sucker, treatment, issue-11]
summary: "OLS panel regressions of contribution on promise/sucker flags and treatment, with two threshold specifications"
status: active
last_verified: "2026-04-19"
---

## Summary

Issue #11 implemented the OLS contribution regression that anchors §4.4-4.5 of the paper. Specification: `contribution ~ made_promise + is_sucker + treatment | round + segment` clustered at session-segment-group. Two columns per liar/sucker threshold (< 20 and < 5).

## Key Coefficients

| Variable | Estimate | Notes |
|---|---|---|
| `treatment` (T2) | **+1.6\*\*\*** | T2 contributes ~1.6 ECU more on average — operates through the **first round** |
| `made_promise` | ~+0.5 | Not significant (ceiling effects: promisers are already contributing high) |
| `is_sucker_20` | **−6.0\*\*\*** | Being suckered drops subsequent contributions by ~6 ECU |
| `is_sucker_5` | **−8.0\*\*\*** | Larger magnitude under stricter threshold |

## Interpretation

- The treatment coefficient flips between this regression (+1.6) and the sentiment-only regression (insignificant, see [sentiment-analysis](sentiment-analysis.md)). That difference is what tells us the treatment effect is concentrated in the **first round** before any chat occurs.
- Promise is insignificant because most people who promise also contribute high — there's no within-promiser variation left after FE absorption.
- The negative `is_sucker` effect motivated the [Sucker DiD Event Study](sucker-did.md) extension.

## Files

- `analysis/contribution_regression.R` — single threshold version.
- `analysis/contribution_regression_combined.R` — combined sentiment + promise/sucker version used in paper Table.
- `output/tables/contribution_regression.tex`, `contribution_regression_combined.tex`.

## Related

- [Behavior Classification](behavior-classification.md)
- [Sentiment Analysis & Regressions](sentiment-analysis.md)
- [Sucker DiD Event Study](sucker-did.md)
