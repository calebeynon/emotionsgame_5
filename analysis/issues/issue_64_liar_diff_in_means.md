# Issue #64: Liar Diff-in-Means by Treatment and Gender

## Summary
Replaces the prior logit regression of lying (from issue #53) with simple Welch's two-sample $t$-tests on the participant-level "ever lied" indicator. Since treatment was randomly assigned, a difference-in-means at the participant level is sufficient.

## Methodology
1. Load `behavior_classifications.csv` and collapse to participant level by taking the max of `lied_this_round_20` within each `(session_code, label)` → `ever_lied` indicator (1 if the participant broke a promise in any round).
2. Load gender from the raw `*_data.csv` survey files (`finalresults.1.player.q1`) and merge on `(session_code, label)`, applying `SESSION_CODE_REMAP` for session 03.
3. Run two Welch two-sample $t$-tests (`scipy.stats.ttest_ind(equal_var=False)`):
   - Treatment 1 vs. Treatment 2 (n=80 each)
   - Male vs. Female (n=73 and n=85; "Other/Prefer not to respond" excluded, n=2)

## Key Results
| Comparison | Mean A | Mean B | Diff (pp) | p-value |
|---|---|---|---|---|
| Treatment 1 vs. Treatment 2 | 42.5% | 26.2% | 16.2 | 0.031 |
| Male vs. Female | 31.5% | 35.3% | -3.8 | 0.617 |

T1 participants are significantly more likely to ever lie than T2 participants. Gender is not a significant predictor.

## Outputs
| Output | Script | Inputs |
|---|---|---|
| `analysis/output/tables/liar_diff_in_means.tex` | `analysis/analysis/issue_64_liar_diff_in_means.py` | `datastore/derived/behavior_classifications.csv`, `datastore/raw/*_data.csv` |

## Paper Changes
- `analysis/paper/Paper.tex`: removed the logit specification (Eq. `eq:liar_logit`) and the `liar_conditional_probability.tex` table block from issue #53; replaced with a short paragraph referencing `\input{liar_diff_in_means.tex}`.

## Notes
- `analysis/analysis/issue_53_liar_regression.R` left untouched per user request; only the paper reference was switched.
- Per-round lie rate was explored but omitted from the final table to keep the comparison simple and focused on the extensive margin.
