# Issue #46: Sentiment-Contribution Regression Analysis

## Overview
Implement regression analysis examining whether chat sentiment directly predicts contribution behavior in the public goods game.

## Deliverables
- R regression script (`analysis/analysis/sentiment_contribution_regression.R`) with two fixest models
- Pytest data validation and integration tests (`analysis/tests/test_sentiment_contribution_regression.py`, 17 tests)
- LaTeX regression table output (`analysis/output/tables/sentiment_contribution_regression.tex`)

## Methodology
- **Baseline**: `contribution ~ sentiment_compound_mean + treatment | round + segment`, clustered SE at session-segment-group
- **Extended**: Same + `message_count` control to isolate sentiment content from chat volume
- Estimation via `fixest::feols()` in R
- Input: `sentiment_scores.csv` (2,298 player-rounds where player sent chat messages)

## Key Results
- Sentiment coefficient ~2.2***, stable across both specifications
- Message count insignificant — sentiment content matters, not volume
- Treatment effect insignificant in this chatting-player subsample
- R-squared ~0.053, typical for behavioral data

## Notes
- Sample conditioned on player-rounds with chat (2,298 of ~3,520 possible obs)
- Sentiment scores are VADER compound, computed at player-round level
- Round 1 excluded from all supergames (no prior chat to generate sentiment)
