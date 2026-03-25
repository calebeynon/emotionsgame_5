# Issue #39: Analyze Emotion-Sentiment Co-Movement and Relationship to Contributions

## Summary
Analyze the relationship between facial emotion valence (from video annotations) and chat sentiment (from VADER analysis), examining how this relationship differs across player types (cooperative state, liar status, sucker status).

## Motivation
Players may present different emotional faces than what their chat text conveys. Understanding this emotion-sentiment gap — especially across cooperative states, liar status, and sucker status — may reveal strategic deception patterns in the public goods game.

## Deliverables
- 4 dot plots comparing z-scored emotion valence vs sentiment compound across player types:
  1. Cooperative state (cooperative vs noncooperative)
  2. Liar status (honest vs liar)
  3. Sucker status (non-sucker vs sucker)
  4. Liar x cooperative state (2x2 interaction)
- Orthogonal decomposition regression (sentiment split into emotion-aligned vs orthogonal components)
- Strategic deception logit model (emotion-sentiment gap predicts noncooperative behavior)
- Gap significance tests (Welch t-tests across player types)
- Shared utilities, master runner, and pytest test suite

## Technical Approach
- Language: R (ggplot2, data.table, fixest)
- Z-score both modalities on complete-case population, compute gap = z(valence) - z(compound)
- Dot plots with 95% CIs using geom_errorbarh + geom_point
- Panel regressions with round + segment FE, clustered SEs at session-segment-group
- Deception logit uses min-max normalized gap as predictor

## Scripts
| Script | Input | Output |
|--------|-------|--------|
| `issue_39_common.R` | `merged_panel.csv`, `behavior_classifications.csv` | (shared utilities) |
| `issue_39_plot_dotplots.R` | merged data | 4 PNG plots |
| `issue_39_gap_tests.R` | merged data | `emotion_sentiment_gap_tests.tex` |
| `issue_39_regression_decomposition.R` | merged data | 3 LaTeX tables |
| `issue_39_all.R` | (sources above) | all outputs |

## Key Results
- Sentiment compound is the stronger predictor of contributions (emotion valence is not significant)
- The emotion-sentiment gap differs significantly by cooperative state (p=0.02), liar status (p=0.01), and sucker status (p=0.002)
- The gap is a highly significant predictor of noncooperative behavior (logit coeff=1.84, p<0.001)
- Liars show much higher facial valence (9.62) than honest players (3.95) but identical chat sentiment (0.128)

## Liar Communication Strategies
GPT-5.4 classified 94 liar chat instances (all with contribution < 20) into communication strategy categories. These capture the heterogeneous ways liars communicate after breaking promises — not just "guilt" but strategic, emotional, and rhetorical patterns:

| Category | Count | % | Description |
|----------|-------|---|-------------|
| false_promise | 60 | 64% | Stated contribution they didn't make |
| deflection_collective | 27 | 29% | "We all should..." diffusion of responsibility |
| no_guilt | 22 | 23% | No strategy-related content detected |
| manipulation | 12 | 13% | Directed others' behavior, rotation schemes |
| blame_shifting | 11 | 12% | Accused others while defecting themselves |
| performative_frustration | 9 | 10% | Acted upset while being a defector |
| self_justification | 8 | 9% | Rationalized own defection |
| duping_delight | 2 | 2% | Appeared amused while deceiving |
| genuine_guilt | 0 | 0% | Sincere apology/remorse |

False promises dominate (64%). Genuine guilt is absent — the 3 hand-coded cases had contribution=25 (cooperating to make amends) and are excluded by the round-specific `lied_this_round_20` flag.
