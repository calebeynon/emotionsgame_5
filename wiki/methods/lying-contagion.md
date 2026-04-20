---
title: "Lying Contagion Regression (Issue #72)"
type: method
tags: [liar, regression, contagion, feedback, treatment]
summary: "Tests whether own lying responds to group lying in prior round/supergame, especially under feedback treatment"
status: active
last_verified: "2026-04-20"
---

## Summary

Tests whether a participant becomes more likely to lie after observing others in their group lie, and whether this "contagion" is amplified by the feedback treatment (Treatment 1). Conceptually a lying analog to the free-riding contagion documented for contributions: complements Table 1 (feedback hurts cooperation) and Table 4 (feedback induces more lies) by asking whether the *mechanism* is a within-group behavioral response to observed lying rather than an independent treatment intercept shift.

## Hypothesis

Under Treatment 1 (feedback), participants see who lied in prior rounds and can condition on it; under Treatment 2 (no feedback) they cannot. If lying is contagious and feedback is the channel, the treatment-by-group-lying interactions should be positive. Key coefficient: `treatment_f=1 × group_lied_*` > 0 → contagion concentrates in the feedback treatment.

## Models

Two regressor specifications × three estimators → six model columns. All columns cluster SEs at the group level (`cluster_group`).

| # | Spec | Estimator | FE | Notes |
|---|------|-----------|----|-------|
| 1 | A (lag) | LPM | segment + round + individual | Individual FE absorbs Treatment main effect |
| 2 | B (cumulative) | LPM | segment + round + individual | Individual FE absorbs Treatment main effect |
| 3 | A (lag) | FE Logit | segment + round + individual | Drops 105 never-liars (N falls to 935) |
| 4 | B (cumulative) | FE Logit | segment + round + individual | Drops 105 never-liars (N falls to 935) |
| 5 | A (lag) | Pooled Logit | segment + round | No individual/session FE; all 160 individuals retained |
| 6 | B (cumulative) | Pooled Logit | segment + round | No individual/session FE; all 160 individuals retained |

**Version A — one-round lag**
```
lied_it = beta_1 group_lied_lag_it + beta_2 self_lied_lag_it
        + beta_3 I(treatment=1) + beta_4 I(treatment=1) * group_lied_lag_it
        + FE + e_it
```

**Version B — cumulative prior**
```
lied_it = gamma_1 any_group_lied_prior_it + gamma_2 any_self_lied_prior_it
        + gamma_3 I(treatment=1) + gamma_4 I(treatment=1) * any_group_lied_prior_it
        + FE + e_it
```

Treatment 2 is the reference level (`relevel(factor(treatment), ref = "2")`), so `i(treatment_f, ref=2)=1` contrasts T1 vs T2.

**Design choices:**
- **No session-clustered SEs** — only 10 sessions, below the ~30–40 rule-of-thumb for cluster-robust asymptotics. We cluster at group level instead.
- **Drop `session_code` from FE** where `label_session` is already present (it's nested, so redundant).
- **Pooled logit uses no individual or session FE** — this is the specific robustness added to avoid the never-liar selection problem of FE logit (which drops 66% of individuals). Tradeoff: without session FE, the Treatment 1 × contagion interaction is identified partly from between-session variation across only 10 sessions.
- **Treatment 1 main effect is absorbed** by `label_session` FE in cols 1–4 (since treatment is individual-invariant). It *is* identified in the pooled logit (cols 5–6).

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
- Layout: 6 columns — `LPM A | LPM B | FE Logit A | FE Logit B | Pooled Logit A | Pooled Logit B`
- All columns cluster SEs at `cluster_group` level
- `etable(..., fitstat = c("n"), se.below = TRUE, tex = TRUE)` with a `dict` mapping coefficients to human labels and a `headers` row for the model-column titles

## Results

The treatment-by-group-lying interaction is **not statistically significant in any of the six columns**. Sign is positive in LPM and FE Logit specs (consistent with contagion hypothesis) but flips negative in pooled logit — sensitive to whether within- or between-session variation drives the interaction (only 10 sessions, 5 per treatment).

| Model | T=1 × contagion coef | SE | p | N |
|---|---|---|---|---|
| (1) LPM A | 0.030 | 0.044 | 0.50 | 2,720 |
| (2) LPM B | 0.045 | 0.043 | 0.30 | 2,720 |
| (3) FE Logit A | 0.370 | 0.836 | 0.66 | 935 |
| (4) FE Logit B | 0.666 | 0.874 | 0.45 | 935 |
| (5) Pooled Logit A | −0.612 | 0.769 | 0.43 | 2,720 |
| (6) Pooled Logit B | −0.201 | 0.683 | 0.77 | 2,720 |

**Key baseline (Treatment 2) coefficients:**
- Pooled Logit A: `group_lied_lag` = 1.240** (SE 0.556) — significant within-group contagion under Treatment 2
- Pooled Logit B: `any_group_lied_prior` = 1.006 (SE 0.549, p = 0.067) — marginal
- Pooled Logit: `self_lied_lag` = 1.69*** and `any_self_lied_prior` = 1.53*** — persistent-liar effect that individual FE absorbs in cols 1–4

**Sample sizes:**
- LPM and Pooled Logit: N = 2,720 (160 individuals × ~17 rounds across 5 supergames, after dropping round 1 of each segment)
- FE Logit: N = 935 — fixest drops 105 never-liar individuals and their 1,785 obs because `label_session` FE is singleton for all-zero outcomes

**Treatment 1 main effect:**
- Cols 1–4: absorbed by `label_session` FE (expected)
- Cols 5–6: identified. Pooled Logit A: β = 0.273 (SE 0.312, p = 0.38); Pooled Logit B: β = 0.320 (SE 0.320, p = 0.32). Positive but not significant.

**Interpretation caveats:**
- Modest power: 2,720 player-rounds, 160 individuals, 10 sessions (5 per treatment), 4 players per group
- Pooled logit's benefit is sample retention (all 160 individuals) at the cost of confounding between-session variation in the interaction
- `any_self_lied_prior` is weakly negative in LPM B (−0.073, p = 0.067) but strongly positive in pooled logit B (1.53***), showing that the within-vs-between individual decomposition matters for the self-persistence pattern

## Testing

- Test file: `analysis/tests/test_issue_72_panel.py` — 23 tests
- Covers: schema and dtypes, round-1 exclusion, lag correctness (single-participant traces through a full segment), sum-minus-self semantics for `group_lied_lag` / `any_group_lied_prior` (including cases where the focal player lied themselves), segment-reset of cumulative flags, session-level row counts, and cluster-id stability
- Run: `cd analysis && uv run pytest tests/test_issue_72_panel.py`

## Related

- [Behavior Classification](behavior-classification.md)
- [Liar Diff-in-Means by Treatment and Gender](liar-diff-in-means.md)
- [Liar Flag: Cumulative vs Round-Specific](liar-flag-comparison.md)
- [Project Glossary](../concepts/glossary.md)
