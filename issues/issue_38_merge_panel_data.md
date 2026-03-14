# Issue #38: Merge Panel Data from State, Sentiment, and Emotion Sources

## Summary
Merge contribution (oTree state), VADER sentiment, and iMotions AFFDEX facial emotion data into a unified panel dataset (`datastore/derived/merged_panel.csv`) for downstream regression and analysis.

## Background
The experiment generates data from three separate sources:
1. **Player state classification** (`datastore/derived/player_state_classification.csv`) -- cooperative state, behavior, promises at session_code/segment/round/label level (3520 rows)
2. **Sentiment scores** (`datastore/derived/sentiment_scores.csv`) -- VADER chat sentiment at session_code/segment/round/label level (2298 rows, rounds 2+ only)
3. **Facial emotion data** (`datastore/Rwork/all.csv`) -- iMotions AFFDEX emotions at page level per session/round/participant (9078 rows)

Key challenges:
- `all.csv` uses session numbers (1, 3-11) and composite IDs like `A3` (label + session number), requiring a mapping to oTree session codes
- `all.csv` has duplicate rows per participant per page from multiple iMotions recording segments, resolved by averaging non-zero rows
- Output keeps page-level granularity (Contribute/Results/ResultsOnly) with sentiment/state data joined at round level
- `all_instructions` annotation included as separate instruction-phase rows

## Approach
- `session_mapping.py`: Maps iMotions session numbers (1, 3-11) to oTree session codes and treatments, parses annotation strings into segment/round/page_type
- `load_emotion_data.py`: Loads raw iMotions AFFDEX emotion CSV, deduplicates multi-segment recordings by averaging non-zero rows, outputs clean emotion data keyed by session_code/label/segment/round/page_type
- `merge_panel_data.py`: Cross-joins state data with 3 page types, appends instruction rows, LEFT JOINs sentiment and emotion data to produce the unified panel
- `test_merge_panel_data.py`: Comprehensive test suite (745 lines) covering annotation parsing, deduplication, session mapping, merge correctness, and edge cases

## Files Changed
- `analysis/derived/load_emotion_data.py` (new, 150 lines)
- `analysis/derived/merge_panel_data.py` (new, 198 lines)
- `analysis/derived/session_mapping.py` (new, 125 lines)
- `analysis/tests/test_merge_panel_data.py` (new, 745 lines)

## Expected Output
`datastore/derived/merged_panel.csv` -- 10,683 rows x 34 columns:
- 10,560 game rows (3520 player-rounds x 3 page types)
- 123 instruction-phase rows
- Key columns: session_code, treatment, segment, round, group, label, page_type
- Merged columns from all three sources (state, sentiment, emotion)

## Related Issues
- Depends on #31 (player-level cooperative state classification)
- Depends on #33 (summary statistics)
