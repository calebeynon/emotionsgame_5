---
title: "Facial Valence Regressions for Liars and Suckers"
type: method
tags: [facial-emotion, affdex, valence, liar, sucker, issue-52]
summary: "Issue #52: AFFDEX facial valence regressed on liar and sucker flags, two face windows"
status: active
last_verified: "2026-04-19"
---

## Summary

Tests whether liars and suckers exhibit detectable facial expression differences using AFFDEX continuous valence. Two windows: the **post-outcome Results page** (within-round) and the **pre-decision Chat page** (lagged forward from previous round's Results page). Final result: liars show pre-decision tension and post-outcome "duper's delight"; suckers show no facial signature.

## Final Specifications

```r
emotion_valence ~ lied_this_round_20 + i(round) | segment + player_id
emotion_valence ~ suckered_this_round + i(round) | segment + player_id
cluster = ~player_id
```

Two-way clustered SEs (player + group-segment-round) in the published table.

## Headline Coefficients

| Window | Lied | Suckered |
|---|---|---|
| Results Page Face (post-outcome) | **+5.487\*\*** (2.306) | 1.885 (1.520) |
| Pre-Decision Chat Face (lagged) | **−2.140\*\*** (0.898) | −1.235 (1.214) |

N = 2,696 (Results) and N = 2,161 (Pre-Decision). The "Pre-Decision" face is the AFFDEX valence on the Results/chat page at the **end of round t-1**, shifted forward within `(player, segment)` to align with the round-t contribution it influenced (see [Chat-Round Pairing](../concepts/chat-round-pairing.md)).

## Extension: All 13 Emotions

`issue_52_gap_regressions_all_emotions.R` runs the same spec across joy, engagement, sentimentality, neutral, fear, etc. Same pattern as valence: positive emotions go up post-outcome and down pre-decision for liars. Suckers are flat across the board — consistent with Paan's observation that suckered players are not visibly upset.

## Methodology Notes

- The simplified face-only spec is the **final form** after experimenting with stacked-panel formulations (face vs chat sentiment, face vs quiz-baseline). Baseline-anchored versions produced coefficients within 0.1 of the face-only version, so the baseline leg was dropped for interpretability.
- "Lied" definition: contribution at least 20 tokens below the player's most recent chat promise.
- "Suckered" definition: a groupmate lied in round t, the player contributed 25, and the player did not themselves lie.

## Files

- `analysis/issue_52_gap_regressions.R` — main 4-column table.
- `analysis/issue_52_gap_regressions_all_emotions.R` — 13-emotion supplementary tables.
- `output/tables/issue_52_valence_sentiment_gap_regressions.tex` — Table 4 in paper.
- `output/tables/issue_52_gap_summary_lied.tex`, `issue_52_gap_summary_suckered.tex` — 13-emotion summaries.
- `tests/test_issue_52_gap.py` — 25 tests pinning coefficients.
- `tests/test_issue_52_all_emotions_gap.py` — 66 tests on the 13-emotion tables.

## Related

- [Chat-Round Pairing Semantics](../concepts/chat-round-pairing.md)
- [Merged Panel Construction](merged-panel.md)
- [Main Paper Overview](../papers/main-paper.md)
