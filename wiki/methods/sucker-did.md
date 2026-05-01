---
title: "Sucker DiD Event Study"
type: method
tags: [did, event-study, sucker, regression, issue-20, issue-59]
summary: "Event-study DiD around the round a player gets suckered, with heterogeneous treatment effects by IF/AF"
status: active
last_verified: "2026-05-01"
---

## Summary

Estimates the dynamic effect of being "suckered" (contributing 25 while a groupmate broke a promise) on subsequent contributions. Issue #20 introduced the pooled event study; issue #59 extended it to allow separate trajectories for IF and AF in a single fully-interacted model.

## Pooled Spec (Issue #20)

```
contribution ~ i(tau, got_suckered, ref = c(0, 999))
             + treatment | round + segment
cluster = cluster_id
```

`tau` is event time (rounds since being suckered). The reference categories `c(0, 999)` omit the event round and the never-treated. Negative `tau` coefficients identify pre-trends; positive `tau` identifies post-event response.

## Heterogeneous Spec (Issue #59)

```
contribution ~ i(tau, suckered_t1, ref = c(0, 999))
             + i(tau, suckered_t2, ref = c(0, 999))
             + treatment | round + segment
cluster = cluster_id
```

Single regression interacting `got_suckered` with treatment dummies — produces separate event-study coefficients per treatment, enabling formal cross-treatment tests.

## Two Thresholds × Two Samples

| Threshold | Sample | Description |
|---|---|---|
| < 20 | Main | Suckered = groupmate contributed < 20 after promising; controls = all non-suckered |
| < 20 | Robust | Same threshold; controls restricted to always-cooperators |
| < 5 | Main / Robust | Stricter threshold — fewer events identified |

## Pipeline

1. `derived/issue_20_build_did_panel.py` → `datastore/derived/issue_20_did_panel.csv`
2. `analysis/issue_59_het_did_regression.R` → `output/tables/issue_59_het_did_contribution.tex`
3. `analysis/issue_59_het_did_coefplot.R` → 4 PDF coefplots (`issue_59_het_did_coefplot_{20,5}_{main,robust}.pdf`)
4. `analysis/issue_20_did_*.R` retains the pooled versions and the coefplots used in the paper appendix.

## Headline Result

- Pre-event coefficients are not significantly different from zero (no anticipation, parallel pre-trends).
- Post-event coefficients are significantly negative in every round and **monotonically decrease** — suckered players keep cutting contributions over time, suggesting a snowball/trust-erosion effect.
- IF vs AF difference is visible in the heterogeneous coefplot (`issue_59_het_did_coefplot_20_main.pdf`) used as Figure in §4.5.

## Identification Assumptions (stated in paper)

1. **Parallel trends** — pre-event coefficients support this empirically.
2. **No anticipation** — credible because participants don't know who will break promises, and groups are reshuffled between segments so no cross-segment learning about specific players.

## Related

- [Behavior Classification](behavior-classification.md)
- [Main Paper Overview](../papers/main-paper.md)
