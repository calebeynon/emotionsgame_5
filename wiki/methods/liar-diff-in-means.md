---
title: "Liar Diff-in-Means by Treatment and Gender (Issue #64)"
type: method
tags: [diff-in-means, t-test, liar, treatment, gender, issue-64]
summary: "Welch t-tests on participant-level 'ever lied' indicator — replaces earlier logit (issue #53)"
status: active
last_verified: "2026-04-19"
---

## Summary

Replaced the earlier issue #53 logit regression of round-level lying with a much simpler participant-level Welch's two-sample t-test on an `ever_lied` indicator. Justified because treatment was randomly assigned, so a difference in means at the participant level is the cleanest inference.

## Method

1. Collapse `behavior_classifications.csv` to participant level: `ever_lied = max(lied_this_round_20)` over all rounds within `(session_code, label)`.
2. Merge gender from raw `*_data.csv` (`finalresults.1.player.q1`), applying `SESSION_CODE_REMAP` for session 03.
3. Two Welch t-tests via `scipy.stats.ttest_ind(equal_var=False)`:
   - T1 vs T2 (n = 80 each).
   - Male vs Female (n = 73 vs 85; "Other/Prefer not to respond" excluded, n = 2).

## Results

| Comparison | Mean A | Mean B | Diff (pp) | p |
|---|---|---|---|---|
| T1 vs T2 | 42.5% | 26.2% | 16.2 | 0.031 |
| Male vs Female | 31.5% | 35.3% | −3.8 | 0.617 |

Treatment matters; gender does not.

## Files

- Script: `analysis/issue_64_liar_diff_in_means.py`
- Output: `output/tables/liar_diff_in_means.tex`
- Paper change: removed `eq:liar_logit` and the `liar_conditional_probability.tex` block, replaced with this table.

## Note

`analysis/issue_53_liar_regression.R` was left untouched per user request. Only the paper reference was switched. If you're touching the regression, prefer the diff-in-means version unless you have a specific reason to fall back to the logit.

## Related

- [Behavior Classification](behavior-classification.md)
- [Main Paper Overview](../papers/main-paper.md)
