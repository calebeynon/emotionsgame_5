# Issue #54: Embedding-Emotion Regressions

## Summary
Regress facial emotion outcomes on chat embedding projection scores. Panel regressions using fixest in R, with player and segment fixed effects, clustered SEs at the group level.

## Specification
```
emotion_Y ~ proj_X + word_count + sentiment_compound_mean | player_id + segment
```
Combined model includes all 4 projections simultaneously.

## Design Decisions
1. **Projection variants**: `_pr_dir_small` only (matches existing pattern)
2. **Liar axis**: Round-level (`proj_rliar_pr_dir_small`) only
3. **Model spec**: Both univariate (1 projection per model) AND combined (all 4 projections)
## Emotion DVs
- `emotion_valence`
- `emotion_joy`
- `emotion_anger`
- `emotion_contempt`
- `emotion_surprise`

## Projection Columns
| Display Name | Column |
|---|---|
| Cooperative | `proj_pr_dir_small` |
| Promise | `proj_promise_pr_dir_small` |
| Homogeneity | `proj_homog_pr_dir_small` |
| Round-liar | `proj_rliar_pr_dir_small` |

## Controls
- `word_count`
- `sentiment_compound_mean`

## Fixed Effects
- `player_id` (label x session_code)
- `segment` (supergame)

## Clustering
- `cluster_id` = session_code x segment x group

## Scripts
| Script | Description | Output |
|---|---|---|
| `analysis/analysis/issue_54_embedding_emotion_regression.R` | 5 emotion DVs x (4 univariate + 1 combined) | `output/tables/issue_54_emotion_*.tex` |

## Data Source
- Input: `datastore/derived/merged_panel.csv` (~10,683 rows)
- Filter: `page_type == "Contribute"` (~2,298 rows)

## Notes
- Emotion measures are from facial expression analysis during the Contribute page
- Projections are cosine similarity scores of chat embeddings along semantic axes
- N = ~1,791 player-rounds after complete-case filter on projections, emotions, and controls
- Cooperative and Homogeneity projections are highly collinear (r=0.98, VIF ~31); combined model coefficients for these two are unreliable — univariate models are the trustworthy specification
