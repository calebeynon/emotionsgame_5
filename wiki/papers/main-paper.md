---
title: "Main Paper: Facial Emotions vs Verbal Sentiments"
type: paper
tags: [paper, results, latex, overleaf]
summary: "Structure, key tables, and key claims of analysis/paper/Paper.tex"
status: active
last_verified: "2026-05-01"
---

## Summary

`analysis/paper/Paper.tex` is the working draft. Title: "Facial Emotions vs Verbal Sentiments in a Public Goods game" (Eynon, Jindapon, Khadka, Razzolini). Two-treatment between-subjects design comparing **Individual Feedback (IF)** vs **Aggregate Feedback (AF)** information regimes in a 5-segment public goods game with chat. Combines AFFDEX facial expression data, VADER chat sentiment, LLM-based promise/liar classification, and text embeddings for external validation.

## Section Structure

| Section | Contents | Key Tables/Figures |
|---|---|---|
| 1. Introduction | Motivation: communication × visibility × emotion | — |
| 2. Related Literature | Communication and visibility in PG games | — |
| 3. Experimental Design | Two treatments, 5 segments, 25-token endowment, 0.4 MPCR | — |
| 4.1 Summary Statistics | Mean contribution across all 22 sequential rounds, contribution CDF | `mean_contribution_by_period.png` (issue #67), `contribution_cdf_by_treatment.png` |
| 4.2 Regression Analysis | Treatment means (Paan's new table) then dynamic Arellano-Bond GMM baseline (issues #57, #68) | `treatment_contribution.tex` (Paan, Overleaf-authored), `dynamic_regression_baseline.tex` (4 cols: IF/AF × mean/min-med-max) |
| 4.3 Classifying Behavior | Liar diff-in-means, sentiment distributions | `liar_diff_in_means.tex`, `liar_count_distribution.png`, `sentiment_distribution_if/af.png` |
| 4.4 Communication Sentiment | OLS: contribution ~ sentiment + controls | `contribution_regression_combined.tex` |
| 4.5 Classification Effect | Sucker DiD event study with heterogeneous TE | `issue_59_het_did_coefplot_20_main.pdf` |
| 4.6 Facial Emotions | Liar/sucker valence regressions (issue #52) | `issue_52_valence_sentiment_gap_regressions.tex` |
| 4.7 Dynamic Panel w/ Communication & Emotions | Extension of §4.2 with chat & facial regressors (issue #68) | `dynamic_regression_extended.tex` (12 cols: 4 baselines × {Base, +Chat, +Chat+Facial}) |
| 4.8 Text Embeddings | Centroid projections + Hanaki external validation | `group_contribution_embedding_regression.tex`, `hanaki_external_validation_inv.tex` |
| 5. Conclusion | (placeholder) | — |
| Appendix | Instructions, quiz, screenshots, pooled DiD | `issue_20_did_coefplot_20_main.png`, `issue_20_did_contribution.tex` |

## Key Equations

- `eq:dynamic_reg` — Arellano-Bond two-step GMM in differences with positive/negative peer-deviation dummies and a `round1` dummy. Instruments: lags 2–5 of contribution. Issue #68 dropped the `round2` and `Δ Segment_t` terms to match the coauthor's Stata `xtabond` Table DP1 spec.
- `eq:contribution_sentiment` — `contribution ~ sentiment + treatment + n_messages | round + segment` (clustered SE at session-segment-group).
- `eq:contribution_regression` — `contribution ~ promise + sucker + treatment | round + segment`.
- `eq:did_contribution` — Heterogeneous DiD event study: `contribution ~ Σ τ × suckered × IF + Σ τ × suckered × AF + ... | round + segment`.
- `eq:gap_lied` / `eq:gap_suckered` — `Y ~ Lied + segment×round FE + player FE`, two-way clustered SE.

## Headline Empirical Claims

1. **First-round Individual-Feedback bump**: IF starts higher in round 1 of each segment; contributions converge to ~25 by mid-segment regardless of treatment. (Source: Paper.tex §4.1 — "higher contribution during the very first round in the Individual Feedback (IF) treatment compared to the Aggregate Feedback (AF) treatment".)
2. **Treatment effect on lying**: IF participants are 16.2 pp more likely to ever lie than AF (42.5% vs 26.2%, p = 0.031). Gender is not significant.
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
