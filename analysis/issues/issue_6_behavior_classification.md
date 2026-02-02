# Issue #6: Behavior Classification (Liar and Sucker Flags)

## Issue Summary

Add sucker and liar classification to the promise dataset. These behavioral flags identify players who broke promises (liars) and players who trusted promise-breakers (suckers), enabling analysis of trust dynamics in the public goods game.

## Definitions

### Liar
A player who makes a promise but contributes below the threshold.
- **High threshold (< 20)**: Contribution < 20 is a broken promise
- **Low threshold (< 5)**: Contribution < 5 is a broken promise

### Sucker
A player who contributes the maximum (25) when a group member broke their own promise.
- Requires the group member to have made AND broken a promise
- Non-chatters can be suckers if they contributed 25 while a group member lied

### Thresholds
- **High threshold (< 20)**: More players qualify as liars/suckers (any contribution below 20 breaks the promise)
- **Low threshold (< 5)**: Fewer players qualify as liars/suckers (only very low contributions break the promise)

### Persistence Rules
- Flags are set in the round AFTER the triggering event occurs
- Round 1 always has False for all flags (no prior history)
- Once flagged as liar or sucker, the flag persists for remaining rounds in the segment
- Flags reset at the start of each new segment (supergame)

## Implementation Approach

New script `classify_behavior.py` that builds on the existing `promise_classifications.csv` output. The script:

1. Loads experiment data from raw session files
2. Loads promise classifications to determine who made promises
3. Iterates through all player-rounds in supergame segments
4. Computes liar flags by checking if player broke a promise in prior rounds
5. Computes sucker flags by checking if player contributed 25 when a group member broke a promise
6. Outputs combined classification data with all behavioral flags

## Files Created/Modified

### Created
- `analysis/classify_behavior.py` - Main classification script with data loading, lookup building, and flag computation
- `analysis/behavior_helpers.py` - Reusable helper functions module providing:
  - Promise data loading utilities
  - Player-round record building functions
  - Group membership lookup
  - DataFrame-based liar/sucker flag computation (alternative API)
- `analysis/tests/test_behavior_classification.py` - Unit tests (34 test cases)

### Dependencies
- `analysis/experiment_data.py` - Experiment data loading framework
- `analysis/datastore/derived/promise_classifications.csv` - Promise classification input

## Output

### File: `analysis/datastore/derived/behavior_classifications.csv`

**Row count**: 3520 (10 sessions x 16 players x 22 rounds across supergames)

### Columns

| Column | Type | Description |
|--------|------|-------------|
| session_code | string | Unique session identifier |
| treatment | int | Treatment group (1 or 2) |
| segment | string | Supergame name (supergame1-5) |
| round | int | Round number within segment |
| group | int | Group ID in subsession |
| label | string | Player label (A-R, skipping I/O) |
| participant_id | int | Participant ID |
| contribution | float | Player's contribution (0-25) |
| payoff | float | Player's payoff for the round |
| made_promise | bool | Whether player made a promise this round |
| is_liar_20 | bool | Liar flag under high threshold (< 20) |
| is_liar_5 | bool | Liar flag under low threshold (< 5) |
| is_sucker_20 | bool | Sucker flag when group member broke high threshold (< 20) |
| is_sucker_5 | bool | Sucker flag when group member broke low threshold (< 5) |

## Testing

### Unit Tests (34 test cases)
Located in `analysis/tests/test_behavior_classification.py`

Test categories:
- **Threshold functions**: `is_promise_broken_20`, `is_promise_broken_5`
- **Round 1 behavior**: All flags are False in round 1
- **Liar classification**: Promise + low contribution = liar
- **Liar persistence**: Flag persists across rounds within segment
- **Liar reset**: Flag resets at new segment
- **Sucker classification**: Contributed 25 when group member lied
- **Sucker persistence**: Flag persists across rounds within segment
- **Sucker reset**: Flag resets at new segment
- **Edge cases**: Non-chatters, boundary contributions, no-promise scenarios
- **Combined classification**: All four flags computed correctly

### Running Tests
```bash
cd /Users/caleb/Research/emotionsgame_5/analysis
uv run pytest tests/test_behavior_classification.py -v
```

## Related Issues

- Issue #2: Promise Classification - provides input data for this analysis
- Used for regression analysis of cooperation behavior and trust dynamics
