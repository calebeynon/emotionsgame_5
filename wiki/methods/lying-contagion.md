---
title: "Lying Contagion Regression (Issue #72)"
type: method
tags: [liar, regression, contagion, feedback, treatment]
summary: "Pooled logit of own lying on groupmates' prior lying, interacted with treatment; reports joint Wald test of total IF effect"
status: active
last_verified: "2026-05-01"
---

## Summary

Tests whether a participant becomes more likely to lie after observing others in their group lie, and whether this "contagion" is amplified by the individual-feedback treatment (IF). Conceptually a lying analog to the free-riding contagion documented for contributions: complements Table 1 (feedback hurts cooperation) and Table 4 (feedback induces more lies) by asking whether the *mechanism* is a within-group behavioral response to observed lying rather than an independent treatment intercept shift.

## Hypothesis

Under IF (individual feedback), participants see who lied in prior rounds and can condition on it; under AF (aggregate feedback) they cannot. If lying is contagious *and* individual feedback is the channel, the treatment-by-group-lying interactions should be positive. Key coefficient: `treatment_f=1 × group_lied_*` > 0 → contagion concentrates in IF.

## Models

Two regressor specifications × pooled logit → two columns. Both columns cluster SEs at the group level (`cluster_group`).

| # | Spec | Estimator | FE | Notes |
|---|------|-----------|----|-------|
| 1 | A (lag) | Pooled Logit | segment + round | No individual/session FE; all 160 individuals retained |
| 2 | B (cumulative) | Pooled Logit | segment + round | No individual/session FE; all 160 individuals retained |

**Version A — one-round lag**
```
lied_it = β₁ group_lied_lag_it + β₂ self_lied_lag_it
        + β₃ I(treatment=1) + β₄ I(treatment=1) × group_lied_lag_it
        + α_segment + α_round + e_it
```

**Version B — cumulative prior**
```
lied_it = γ₁ any_group_lied_prior_it + γ₂ any_self_lied_prior_it
        + γ₃ I(treatment=1) + γ₄ I(treatment=1) × any_group_lied_prior_it
        + α_segment + α_round + e_it
```

AF (treatment code `2`) is the reference level (`relevel(factor(treatment), ref = "2")`), so `i(treatment_f, ref=2)=1` contrasts IF vs AF. `β₁` and `γ₁` therefore identify the groupmate-lying effect under AF, and `β₄`, `γ₄` capture the IF differential.

**Design choices:**
- **Pooled logit, no individual or session FE** — avoids the never-liar selection problem of FE logit (which drops 66% of individuals). Tradeoff: identification of the IF × contagion interaction pools within- and between-session variation across only 10 sessions.
- **No session-clustered SEs** — only 10 sessions, below the ~30–40 rule-of-thumb for cluster-robust asymptotics. Cluster at group level (`session × segment × group_id`) instead.
- **Group-contagion regressors exclude the focal player** ("sum-minus-self" arithmetic), so `group_lied_*` captures *other* group members' lying, not the player's own.

### Joint Wald test

At the bottom of the table, the R script reports the Wald test of
```
H0: β₁ + β₄ = 0    (Model A)
H0: γ₁ + γ₄ = 0    (Model B)
```
i.e. zero **total** groupmate-lying effect under IF. Test statistic is $\chi^2_1$; computed from the clustered vcov via the delta method:
```r
k = e_main + e_interaction       # contrast vector
est = k' β̂
var = k' V k                     # V = clustered vcov
χ² = est² / var
```

## Data

- Source panel: `datastore/derived/issue_72_panel.csv` (2,720 rows, 13 columns)
- Builder: `derived/build_issue_72_panel.py` — reads `behavior_classifications.csv`, drops round 1 of each segment (no lag available), applies session-code remap
- `add_lied()` enforces a dtype/value whitelist: bool, integer ⊆ {0,1}, or object ⊆ {True, False, 'True', 'False'}. NaN and any other dtype/value raise `ValueError` rather than silently coercing to 0
- Regression driver: `analysis/analysis/issue_72_lying_contagion_regression.R`
- Sample: 160 individuals × ~17 round-segments = 2,720 obs

## Variables

| Column | Type | Definition |
|---|---|---|
| `session_code` | str | oTree session identifier (10 sessions, 5 per treatment) |
| `treatment` | int | 1 = IF (individual feedback), 2 = AF (aggregate feedback) |
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
| `label_session` | str | `label_session_code` — stable individual id (unused in the 2-column spec) |

## Output

- LaTeX table: `output/tables/issue_72_lying_contagion.tex`
- Layout: 2 columns — `Pooled Logit A | Pooled Logit B`
- All columns cluster SEs at `cluster_group` level
- `etable(..., fitstat = c("n"), se.below = TRUE, tex = TRUE)` with `extralines` adding the joint-Wald `χ²` and `p-value` rows in the fit-statistics section

## Results

### Main coefficients (AF baseline — main effects)

| Model | Group-lying coef | SE | p | N |
|---|---|---|---|---|
| (1) Pooled Logit A (lag) | **1.240** | 0.556 | 0.026 | 2,720 |
| (2) Pooled Logit B (cumulative) | 1.006 | 0.549 | 0.067 | 2,720 |

Under the AF baseline, a groupmate lying in the prior round is associated with elevated log-odds of own lying (1.24, significant at 5%); the cumulative-prior version is marginal (p = 0.067).

### IF × group-lying interaction

| Model | Interaction coef | SE | p |
|---|---|---|---|
| (1) Pooled Logit A (lag) | −0.612 | 0.769 | 0.43 |
| (2) Pooled Logit B (cumulative) | −0.201 | 0.683 | 0.77 |

### Joint Wald test — total IF effect (β_main + β_interaction = 0)

| Model | Sum est | SE | Wald χ² | p |
|---|---|---|---|---|
| (1) Pooled Logit A (lag) | 0.628 | 0.578 | 1.182 | 0.277 |
| (2) Pooled Logit B (cumulative) | 0.805 | 0.597 | 1.817 | 0.178 |

The negative interaction roughly cancels the positive main effect in both specs. The total groupmate-lying response under IF is not distinguishable from zero at conventional levels. The significant contagion in column (1) is therefore an AF phenomenon.

### Other coefficients

- `self_lied_lag` = 1.69*** and `any_self_lied_prior` = 1.53*** — strong persistent-liar effect (identified in pooled logit; would be absorbed by individual FE).
- IF main effect: Pooled Logit A: β = 0.273 (SE 0.312, p = 0.38); Pooled Logit B: β = 0.320 (SE 0.320, p = 0.32). Positive but not significant.

### Interpretation caveats

- Modest power: 2,720 player-rounds, 160 individuals, 10 sessions (5 per treatment), 4 players per group
- Pooled logit's benefit is sample retention (all 160 individuals) at the cost of confounding between-session variation in the interaction
- Prior versions also reported LPM and FE-logit columns; those were dropped to keep the table focused on the comparison the Wald test makes interpretable

## Testing

- Test file: `analysis/tests/test_issue_72_panel.py` — 26 tests
- Manually-traced case constants live in `analysis/tests/fixtures/issue_72_cases.py`
- Covers: schema and dtypes (including binary-column integer-dtype enforcement), round-1 exclusion, lag correctness (single-participant traces through a full segment), sum-minus-self semantics for `group_lied_lag` / `any_group_lied_prior` (including cases where the focal player lied themselves), segment-reset of cumulative flags, session-level row counts, and cluster-id stability within `(session, segment, group)`
- Run: `cd analysis && uv run pytest tests/test_issue_72_panel.py`

## Paper

- Table included in `analysis/paper/Paper.tex` right after the liar-distribution figure (search `\label{table:lying_contagion}`)
- Paper text surrounding the table contains the two specification equations, variable definitions, and a minimal factual caption; no interpretive prose in the paper

## Related

- [Behavior Classification](behavior-classification.md)
- [Liar Diff-in-Means by Treatment and Gender](liar-diff-in-means.md)
- [Liar Flag: Cumulative vs Round-Specific](liar-flag-comparison.md)
- [Project Glossary](../concepts/glossary.md)
