# Issue #52: Facial Valence Regressions for Liars and Suckers

## Summary

Analyze whether players who lie (under-contribute relative to their stated chat intention) or get suckered (contribute fully while a groupmate lies) show differential facial valence (AFFDEX) patterns — both before and after contribution decisions.

## Motivation

Prior work established that liars show detectable emotional signals in chat text sentiment. This issue asks whether facial expressions carry complementary or independent information: do liars exhibit pre-decision tension or post-outcome "duper's delight"? And do suckers visibly react when they discover they've been taken advantage of?

## Final Regression Spec

```r
emotion_valence ~ lied_this_round_20 + i(round) | segment + player_id
cluster = ~player_id
```

Estimated separately for the `suckered_this_round` flag. Two face windows:
- **Results Page Face**: AFFDEX valence on the post-outcome `ResultsOnly` page (N=2,696)
- **Pre-Decision Chat Face**: AFFDEX valence on the end-of-round-$t{-}1$ `Results`/chat page, lagged forward within player-segment to align with round-$t$ contribution (N=2,161)

## Key Findings

| | Lied | Suckered |
|---|---|---|
| Results Page Face | **+5.487\*\*** (2.306) | 1.885 (1.520) |
| Pre-Decision Chat Face | **−2.140\*\*** (0.898) | −1.235 (1.214) |

- **Pre-lie tension → duper's delight**: Liars show suppressed positive affect during pre-contribution chat and elevated valence on the results page after the outcome lands.
- **No sucker signature**: Suckers show no significant facial response in any specification — consistent with Paan's original observation that suckered players are not visibly upset.
- **All-emotions extension confirms pattern**: The same pattern holds for joy (+7.63/−1.87), engagement (+7.09/−3.34), sentimentality (+2.00/−0.69), and in the opposite direction for neutral (−7.71/−2.01). Fear is suppressed on results pages of lying rounds (−0.36).

## Methodological Notes

- **Identification**: The simplified face-only spec (player + segment fixed effects, player-clustered SE) is the final form after experimenting with stacked-panel formulations (face vs chat-sentiment, face vs quiz-baseline). The baseline-anchored versions produced coefficients within 0.1 of the face-only version, confirming the baseline leg contributes almost nothing to point identification.
- **Lied definition**: Contribution is at least 20 tokens below the player's most recent chat promise.
- **Suckered definition**: A groupmate lied in round $t$, the player contributed the full 25-token endowment, and the player did not themselves lie.
- **Chat-round pairing**: The chat on the Results page of round $t{-}1$ is the chat that preceded the round-$t$ contribution. The `shift` operation within player-segment pairs face observations with the contribution they influenced.

## Files Modified / Added

- `analysis/analysis/issue_52_gap_regressions.R` — main headline regressions (152 LOC)
- `analysis/analysis/issue_52_gap_regressions_all_emotions.R` — supplementary 13-emotion extension (237 LOC)
- `analysis/output/tables/issue_52_valence_sentiment_gap_regressions.tex` — final 4-column Table 4
- `analysis/output/tables/issue_52_gap_summary_lied.tex` — 13-emotion summary (Lied)
- `analysis/output/tables/issue_52_gap_summary_suckered.tex` — 13-emotion summary (Suckered)
- `analysis/tests/test_issue_52_gap.py` — 25 tests pinning coefficients and table structure
- `analysis/tests/test_issue_52_all_emotions_gap.py` — 66 tests covering the 13-emotion tables
- `analysis/paper/Paper.tex` — Table 4 block with equations, variable definitions, and caption

## Exploratory Path (Superseded)

Earlier commits on this branch explored:
1. Chat-text sentiment counterfactual (stacked DiD: face vs VADER compound)
2. Quiz-baseline counterfactual (stacked DiD: face vs per-player neutral resting face from `introduction/all_instructions`)
3. Post-results chat face as a third spec
4. Combined-interaction (both flags in one regression)

These were iteratively simplified away after confirming the baseline leg contributes ~0.1 of coefficient movement and the chat-text counterfactual didn't add explanatory power. The final face-only spec with player + segment FE is numerically almost identical to the stacked versions and much easier to interpret.
