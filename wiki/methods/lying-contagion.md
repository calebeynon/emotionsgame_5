---
title: "Lying Contagion Regression (Issue #72)"
type: method
tags: [liar, regression, contagion, feedback, treatment]
summary: "Tests whether own lying responds to group lying in prior round/supergame, especially under feedback treatment"
status: active
last_verified: "2026-04-19"
---

## Summary

Tests whether a participant becomes more likely to lie after observing others in their group lie, and whether this "contagion" is amplified by the feedback treatment (Treatment 1). Conceptually a lying analog to the free-riding contagion documented for contributions: complements Table 1 (feedback hurts cooperation) and Table 4 (feedback induces more lies) by asking whether the *mechanism* is a within-group behavioral response to observed lying rather than an independent treatment intercept shift.

## Hypothesis

Under Treatment 1 (feedback), participants see who lied in prior rounds and can condition on it; under Treatment 2 (no feedback) they cannot. If lying is contagious and feedback is the channel, the treatment-by-group-lying interactions should be positive. Key coefficient: `treatment_f=1 × group_lied_*` > 0 → contagion concentrates in the feedback treatment.

## Models

Two regressor specifications, each reporting LPM with two clusterings (group, session) and a group-clustered logit. Six model columns total.

**Version A — one-round lag**
```
lied_it = beta_1 group_lied_lag_it + beta_2 self_lied_lag_it
        + beta_3 I(treatment=1) + beta_4 I(treatment=1) * group_lied_lag_it
        + FE(session, segment, round, individual) + e_it
```

**Version B — cumulative prior**
```
lied_it = gamma_1 any_group_lied_prior_it + gamma_2 any_self_lied_prior_it
        + gamma_3 I(treatment=1) + gamma_4 I(treatment=1) * any_group_lied_prior_it
        + FE(session, segment, round, individual) + e_it
```

Treatment 2 is the reference level (`relevel(factor(treatment), ref = "2")`), so the `i(treatment_f, ref=2)=1` contrast is T1 vs T2. The `treatment_f=1` main effect is absorbed by `session_code` FE (treatment is constant within session) and fixest drops it automatically — this is expected.

Group-contagion regressors exclude the focal player ("sum-minus-self" arithmetic), so `group_lied_*` captures *other* group members' lying, not the player's own.

## Data

- Source panel: `datastore/derived/issue_72_panel.csv` (2,720 rows, 13 columns)
- Builder: `derived/build_issue_72_panel.py` — reads `behavior_classifications.csv`, drops round 1 of each segment (no lag available), applies session-code remap
- Regression driver: `analysis/analysis/issue_72_lying_contagion_regression.R`
- Sample: 160 individuals × ~17 round-segments = 2,720 obs (LPM); 935 obs for logit after fixest removes all-0/all-1 individual FE groups (105 individuals are singleton for the binomial likelihood)

## Variables

| Column | Type | Definition |
|---|---|---|
| `session_code` | str | oTree session identifier (10 sessions, 5 per treatment) |
| `treatment` | int | 1 = feedback, 2 = no-feedback |
| `segment` | str | `supergame1`..`supergame5` |
| `round` | int | Round within segment (1..N_rounds) |
| `group` | int | Group id within `(session, segment)` |
| `label` | str | Participant label A-R (I/O skipped) |
| `lied` | 0/1 | `lied_this_round_20` — promised AND contributed <20 this round |
| `self_lied_lag` | 0/1 | Focal player's `lied` in round t-1, same segment |
| `group_lied_lag` | 0/1 | Any OTHER group member lied in round t-1 (sum-minus-self) |
| `any_self_lied_prior` | 0/1 | Focal player has ever lied in this segment before round t |
| `any_group_lied_prior` | 0/1 | Any OTHER group member has ever lied in this segment before round t |
| `cluster_group` | str | `session_code_segment_group` — stable group id for clustering |
| `label_session` | str | `label_session_code` — stable individual id for FE |

## Output

- LaTeX table: `output/tables/issue_72_lying_contagion.tex`
- Layout: 6 columns — `LPM A (grp) | LPM A (ses) | LPM B (grp) | LPM B (ses) | Logit A (grp) | Logit B (grp)`
- `etable(..., fitstat = c("n"), se.below = TRUE, tex = TRUE)` with a `dict` mapping coefficients to human labels and a `headers` row for the model-column titles

## Results

Both treatment-by-group-lying interactions are positive (sign consistent with the contagion hypothesis) but not statistically significant at conventional levels.

| Model | Coefficient | Estimate | SE | p |
|---|---|---|---|---|
| m1 LPM A (grp) | `T=1 × group_lied_lag` | 0.030 | 0.044 | 0.50 |
| m2 LPM A (ses) | `T=1 × group_lied_lag` | 0.030 | 0.054 | 0.59 |
| m3 LPM B (grp) | `T=1 × any_group_lied_prior` | 0.045 | 0.043 | 0.30 |
| m4 LPM B (ses) | `T=1 × any_group_lied_prior` | 0.045 | 0.049 | 0.38 |
| m5 Logit A (grp) | `T=1 × group_lied_lag` | 0.370 | 0.840 | 0.66 |
| m6 Logit B (grp) | `T=1 × any_group_lied_prior` | 0.666 | 0.879 | 0.45 |

Sample sizes: LPM N = 2,720 across all four LPM models; logit N = 935 for both logits. The logit drop reflects `label_session` FE absorbing all-0-outcome individuals (non-liars) — fixest removes 105 singleton individuals and 1,785 observations for the binomial likelihood. The `treatment_f=1` main effect is dropped as collinear with `session_code` FE (treatment is constant within session) and is *not* a modelling bug.

Only marginal non-interaction finding: `any_self_lied_prior` is −0.073 (p = 0.067) in m3 — prior own-lying slightly *reduces* current lying at the 10% level, suggesting a weak one-and-done pattern rather than escalation. Null results on the contagion interactions should be read in light of the modest sample (2,720 player-rounds, 160 individuals across 10 sessions); with 4 players per group and only 5 sessions per treatment, between-group statistical power for the interaction is limited.

## Testing

- Test file: `analysis/tests/test_issue_72_panel.py` — 16 tests
- Covers: schema and dtypes, round-1 exclusion, lag correctness (single-participant traces through a full segment), sum-minus-self semantics for `group_lied_lag` / `any_group_lied_prior` (including cases where the focal player lied themselves), segment-reset of cumulative flags, session-level row counts, and cluster-id stability
- Run: `cd analysis && uv run pytest tests/test_issue_72_panel.py`

## Related

- [Behavior Classification](behavior-classification.md)
- [Liar Diff-in-Means by Treatment and Gender](liar-diff-in-means.md)
- [Liar Flag: Cumulative vs Round-Specific](liar-flag-comparison.md)
- [Project Glossary](../concepts/glossary.md)
