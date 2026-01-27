# Issue #11: Contribution Regression Analysis

## Purpose
Implement regression analysis to estimate the effects of promise-making and "sucker" status on contribution behavior in the public goods game, controlling for treatment effects.

## Approach
Fixed effects panel regression using `fixest`:
- **Model**: `contribution ~ made_promise + is_sucker + treatment | round + segment`
- **Clustering**: Standard errors clustered at session-segment-group level
- **Sucker definitions**: Two specifications (strict and lenient) based on promise-breaking thresholds

### Coefficient Interpretation
- **treatment**: Effect of Treatment 2 relative to Treatment 1 (reference)
- **made_promise**: Effect of having made a promise on own contribution
- **is_sucker**: Effect of being "suckered" (contributed max while groupmate broke promise)

### Sucker Classification
- **Strict**: Groupmate contributed < 20 after promising (stricter = more suckers)
- **Lenient**: Groupmate contributed < 5 after promising (lenient = fewer suckers)

## Inputs
| File | Description |
|------|-------------|
| `datastore/derived/behavior_classifications.csv` | Player-round data with promise and sucker flags |

## Outputs
| File | Description | Script |
|------|-------------|--------|
| `output/analysis/contribution_regression.tex` | LaTeX regression table | `contribution_regression.R` |

## Key Results
- **Treatment effect**: ~1.6 points higher contribution in Treatment 2 (p < 0.01)
- **Made promise**: ~0.5 points (not significant)
- **Sucker effect (strict)**: -6.0 points (p < 0.01)
- **Sucker effect (lenient)**: -8.0 points (p < 0.01)

## Key Changes
- Added `analysis/analysis/contribution_regression.R`: Main regression script
- Added `output/analysis/contribution_regression.tex`: LaTeX table output
- Added paper files (`analysis/paper/`) for manuscript integration
- Updated `.gitignore` to exclude LaTeX build artifacts

## Testing
- Validated required columns exist in input data
- Checked for missing values in key variables
- Verified coefficient signs match theoretical expectations
