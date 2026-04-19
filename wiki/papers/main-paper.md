---
title: "Main Paper: Facial Emotions vs Verbal Sentiments"
type: paper
tags: [paper, results, latex, overleaf]
summary: "Structure, key tables, and key claims of analysis/paper/Paper.tex"
status: active
last_verified: "2026-04-19"
---

## Summary

`analysis/paper/Paper.tex` is the working draft. Title: "Facial Emotions vs Verbal Sentiments in a Public Goods game" (Eynon, Jindapon, Khadka, Razzolini). Two-treatment between-subjects design comparing **No Feedback (T1)** vs **Feedback (T2)** information regimes in a 5-segment public goods game with chat. Combines AFFDEX facial expression data, VADER chat sentiment, LLM-based promise/liar classification, and text embeddings for external validation.

## Section Structure

| Section | Contents | Key Tables/Figures |
|---|---|---|
| 1. Introduction | Motivation: communication × visibility × emotion | — |
| 2. Related Literature | Communication and visibility in PG games | — |
| 3. Experimental Design | Two treatments, 5 segments, 25-token endowment, 0.4 MPCR | — |
| 4.1 Summary Statistics | Mean/median contribution by round, contribution CDF | `mean_contribution_by_round.png`, `median_contribution_by_round.png`, `contribution_cdf_by_treatment.png` |
| 4.2 Regression Analysis | Dynamic Arellano-Bond GMM (issue #57) | `dynamic_regression.tex` (3 specs × 2 treatments) |
| 4.3 Classifying Behavior | Liar diff-in-means, sentiment distributions | `liar_diff_in_means.tex`, `liar_count_distribution.png`, `sentiment_distribution_t1/t2.png` |
| 4.4 Communication Sentiment | OLS: contribution ~ sentiment + controls | `contribution_regression_combined.tex` |
| 4.5 Classification Effect | Sucker DiD event study with heterogeneous TE | `issue_59_het_did_coefplot_20_main.pdf` |
| 4.6 Facial Emotions | Liar/sucker valence regressions (issue #52) | `issue_52_valence_sentiment_gap_regressions.tex` |
| 4.7 Text Embeddings | Centroid projections + Hanaki external validation | `group_contribution_embedding_regression.tex`, `hanaki_external_validation_inv.tex` |
| 5. Conclusion | (placeholder) | — |
| Appendix | Instructions, quiz, screenshots, pooled DiD | `issue_20_did_coefplot_20_main.png`, `issue_20_did_contribution.tex` |

## Key Equations

- `eq:dynamic_reg` — Arellano-Bond two-step GMM in differences with positive/negative deviation dummies and round dummies. Instruments: lags 2-5 of contribution.
- `eq:contribution_sentiment` — `contribution ~ sentiment + treatment + n_messages | round + segment` (clustered SE at session-segment-group).
- `eq:contribution_regression` — `contribution ~ promise + sucker + treatment | round + segment`.
- `eq:did_contribution` — Heterogeneous DiD event study: `contribution ~ Σ τ × suckered × T1 + Σ τ × suckered × T2 + ... | round + segment`.
- `eq:gap_lied` / `eq:gap_suckered` — `Y ~ Lied + segment×round FE + player FE`, two-way clustered SE.

## Headline Empirical Claims

1. **First-round Feedback bump**: T2 starts higher in round 1 of each segment; contributions converge to ~25 by mid-segment regardless. The CDF in T1 first-order stochastically dominates T2 (lower contributions overall under No Feedback).
2. **Treatment effect on lying**: T1 participants are 16.2 pp more likely to ever lie than T2 (42.5% vs 26.2%, p = 0.031). Gender is not significant.
3. **Liar prevalence is concentrated**: 105 of 160 participants never lie in 22 rounds.
4. **Sucker → contribution decay**: Being suckered reduces subsequent contributions monotonically (Sucker coefficient ≈ −6, p < 0.01 in OLS; event-study coefficients monotonically more negative post-event).
5. **Sentiment → contribution**: VADER compound sentiment coefficient ≈ +2.18 (p < 0.01); message count not significant. (Sample limited to chatters; round 1 excluded.)
6. **Pre-lie tension, post-lie delight**: Liars' AFFDEX valence is suppressed during pre-decision chat (β = −2.14, p < 0.05) and elevated on the post-outcome Results page (β = +5.49, p < 0.05). Suckers show no facial response.
7. **Embeddings transfer externally**: Centroid projections trained on this experiment's chat predict investment in Hanaki & Ozkes (2023) French bilateral-investment data — Promise (+5.55) and Round Liar (−7.98) project significantly.

## Where Tables and Figures Come From

- All tables in `analysis/paper/Paper.tex` use bare filenames (no directory). Locally they resolve to `../output/tables/`; on Overleaf they resolve to `tables/`. Same for plots.
- The GitHub Action `.github/workflows/sync-overleaf.yml` parses `\input{}` and `\includegraphics{}` and copies referenced files into `analysis/paper/tables/` and `plots/` before pushing to Overleaf.
- **Never commit files manually** to `analysis/paper/tables/` or `analysis/paper/plots/` — the action manages them.

## Author Notes Embedded in Paper

The PDF has yellow callout boxes flagging:
- Dynamic-regression interpretation needs Paan's input.
- External validity section: looking for additional datasets beyond Hanaki & Ozkes.

## Related

- [Dynamic Regression (Arellano-Bond)](../methods/dynamic-regression.md)
- [Sucker DiD Event Study](../methods/sucker-did.md)
- [Facial Valence Regressions](../methods/facial-valence-regressions.md)
- [Embeddings Pipeline & External Validation](../methods/embeddings-validation.md)
- [Behavior Classification](../methods/behavior-classification.md)
