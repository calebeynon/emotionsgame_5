# Issue #17: Sentiment-Contribution Regression with Liar Interaction

**Status:** In Progress
**Branch:** issue_17_sentiment_liar_regression
**Created:** 2026-01-26

## Objective

Implement regression analysis to test whether chat sentiment predicts contribution behavior differently for liars versus non-liars in the public goods game.

## Hypothesis

Sentiment expressed in chat is predictive of contributions for non-liars (who are communicating genuinely), but this relationship is attenuated or null for liars (who may use positive chat strategically without intending to follow through).

## Model Specification

```
Contribution_it = beta_1(Sentiment_it) + beta_2(Liar_i) + beta_3(Sentiment x Liar) + controls + epsilon
```

### Variables
- **Contribution_it**: Player i's contribution in round t (0-25 points)
- **Sentiment_it**: Mean compound sentiment score of chat messages influencing round t's contribution
- **Liar_i**: Binary indicator for whether player has broken a promise in a prior round
- **Sentiment x Liar**: Interaction term testing attenuation hypothesis
- **controls**: Treatment, round, segment fixed effects; clustered standard errors

### Key Coefficients
- **beta_1**: Effect of sentiment on contribution for non-liars (expected positive)
- **beta_2**: Baseline contribution difference for liars (expected negative)
- **beta_3**: Differential effect of sentiment for liars vs non-liars (expected negative, attenuating beta_1)

## Chat-Round Pairing Semantics

**Critical**: Chat from round N-1 influences contribution in round N.

In the oTree experiment, within each round:
1. Player makes contribution decision
2. Results shown
3. Chat occurs (after seeing results)

Therefore:
- Round 1 has no prior chat to influence it (empty chat messages)
- Round N's contribution is influenced by round N-1's chat
- Last round's chat influences no subsequent contribution (orphan chat)

The `sentiment_scores.csv` file already encodes this pairing correctly from the upstream `compute_sentiment.py` script.

## Inputs

| File | Description |
|------|-------------|
| `datastore/derived/sentiment_scores.csv` | Player-round sentiment scores (VADER) |
| `datastore/derived/behavior_classifications.csv` | Player-round liar/sucker flags |

### sentiment_scores.csv Columns

| Column | Description |
|--------|-------------|
| `session_code` | Unique session identifier |
| `treatment` | Treatment group (1 or 2) |
| `segment` | Supergame name (supergame1-5) |
| `round` | Round number within segment |
| `group` | Group ID within round |
| `label` | Player label (A-R, excluding I/O) |
| `participant_id` | Unique participant identifier |
| `contribution` | Player's contribution (0-25) |
| `payoff` | Player's round payoff |
| `message_count` | Number of chat messages from this player |
| `sentiment_compound_mean` | Mean VADER compound score (-1 to 1) |
| `sentiment_compound_std` | Std dev of compound scores across messages |
| `sentiment_compound_min` | Minimum compound score |
| `sentiment_compound_max` | Maximum compound score |
| `sentiment_positive_mean` | Mean positive sentiment component (0-1) |
| `sentiment_negative_mean` | Mean negative sentiment component (0-1) |
| `sentiment_neutral_mean` | Mean neutral sentiment component (0-1) |

### behavior_classifications.csv Columns (relevant subset)

| Column | Description |
|--------|-------------|
| `session_code`, `segment`, `round`, `label` | Join keys |
| `is_liar_20` | True if player broke promise (contributed < 20) in any prior round |
| `is_liar_5` | True if player broke promise (contributed < 5) in any prior round |

## Merge Instructions

Merge `sentiment_scores.csv` with `behavior_classifications.csv` on:
- `session_code`
- `segment`
- `round`
- `label`

This brings liar flags into the sentiment dataset for regression analysis.

## Outputs

| File | Description | Script |
|------|-------------|--------|
| `output/analysis/sentiment_liar_regression.tex` | LaTeX regression table | `sentiment_liar_regression.R` |

## Expected Results Structure

Multiple model specifications:
1. **Baseline**: Sentiment only
2. **Main effects**: Sentiment + Liar
3. **Interaction**: Sentiment + Liar + Sentiment x Liar
4. **Full**: All above with treatment and fixed effects

## Testing

- [ ] Verify merge produces expected row count
- [ ] Check for missing values in key variables
- [ ] Validate sentiment scores are in expected range [-1, 1]
- [ ] Confirm liar flags are boolean
- [ ] Verify coefficient signs match theoretical expectations
