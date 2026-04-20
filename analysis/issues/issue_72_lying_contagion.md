# Issue #72: Lying Contagion Regression

## Summary
Estimates whether observing a groupmate lie in the prior round (or earlier in the supergame) increases a player's own probability of lying in round $t$, and whether this "contagion" differs by feedback treatment. Joint Wald test at the bottom of the regression table tests whether the *total* Treatment-1 response equals zero.

## Methodology
1. Build a player-round panel (`datastore/derived/issue_72_panel.csv`) from `behavior_classifications.csv` via `analysis/derived/build_issue_72_panel.py`. Drops round 1 of each segment (no lag available). Computes sum-minus-self group-lying indicators.
2. Run two pooled logit specifications with segment + round FE, clustering SEs at the group level:
   - **Version A (one-round lag):** key regressor `group_lied_lag` (any other groupmate lied in round $t{-}1$), interacted with treatment
   - **Version B (cumulative prior):** key regressor `any_group_lied_prior` (any other groupmate lied in any earlier round of the current supergame), interacted with treatment
3. Treatment 2 is the reference level, so the main group-lying coefficient identifies the effect under T2 and the interaction captures the T1-vs-T2 differential.
4. Compute the joint Wald test $H_0: \beta_{\text{main}} + \beta_{\text{main} \times T=1} = 0$ from the clustered vcov (chi-squared with 1 df). This tests whether the **total** groupmate-lying effect under Treatment 1 equals zero.

## Key Results
| Model | T2 main coef | SE | T1 interaction | Wald χ² (T1 total = 0) | p |
|---|---|---|---|---|---|
| (1) Logit A (lag) | **1.240** | 0.556 | −0.612 (n.s.) | 1.182 | 0.277 |
| (2) Logit B (cumulative) | 1.006 | 0.549 | −0.201 (n.s.) | 1.817 | 0.178 |

- Under Treatment 2 (no feedback), the one-round-lag groupmate-lying effect is 1.240 (p = 0.026).
- Under Treatment 1 (feedback), the negative interaction roughly cancels the main effect; the joint Wald test cannot reject zero total response.
- Observed within-group contagion is a **Treatment-2 phenomenon**; the feedback treatment does not amplify contagion on this pooled-logit specification.

## Outputs
| Output | Script | Inputs |
|---|---|---|
| `datastore/derived/issue_72_panel.csv` | `analysis/derived/build_issue_72_panel.py` | `datastore/derived/behavior_classifications.csv` |
| `analysis/output/tables/issue_72_lying_contagion.tex` | `analysis/analysis/issue_72_lying_contagion_regression.R` | `datastore/derived/issue_72_panel.csv` |

## Paper Changes
- `analysis/paper/Paper.tex`: added two specification equations (Version A and Version B), variable definitions, and the table input after the liar-distribution figure. Minimal factual caption; no interpretive prose (user writes their own narrative).

## Wiki Changes
- `wiki/methods/lying-contagion.md`: updated to reflect 2-column spec (Logit A, Logit B), documents the joint Wald test computation (contrast vector on clustered vcov), refreshed results tables.

## Tests
- `analysis/tests/test_issue_72_panel.py` — 23 tests covering schema/dtypes, round-1 exclusion, lag correctness (single-participant traces), sum-minus-self semantics for `group_lied_lag` / `any_group_lied_prior`, segment-reset of cumulative flags, session-level row counts, cluster-id stability.
- Run: `cd analysis && uv run pytest tests/test_issue_72_panel.py`

## Notes
- Prior versions of the table also reported LPM and FE-logit columns; those were dropped in this PR to keep the table focused on the specification that retains all 160 individuals (pooled logit) and makes the joint Wald test interpretable.
- Column headers are `Logit A` / `Logit B` (not `Pooled Logit A/B`) for brevity.
- 10 sessions, 5 per treatment — cluster-robust asymptotics at the session level would be marginal; clustering at the group level (`session × segment × group_id`) is used instead.
