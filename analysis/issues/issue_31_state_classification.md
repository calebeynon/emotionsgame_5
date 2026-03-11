# Issue #31: Cooperative State Classification

**Status:** Complete
**Branch:** issue_31_state_classification
**Created:** 2026-02-16

## Objective

Classify each group-round (and each player-round) into cooperative vs noncooperative states, then cross-reference with player behavior and promise-making to produce a 2x2 matrix (behavior x promise) within each state.

## Two Classification Approaches

### Group-Level (`classify_states.py`)
- A group-round is **cooperative** if the group mean contribution >= 75% of endowment (25)
- All 4 players in that group-round share the same state
- Output: `datastore/derived/state_classification.csv`

### Player-Level (`classify_player_states.py`)
- Each player's state is determined by the **other 3 players'** contributions
- If others' total contributions >= 60, the player is in a cooperative world
- Two players in the same group can have different state classifications
- Output: `datastore/derived/player_state_classification.csv`

## Architecture

```
analysis/derived/
  classify_states.py          # Group-level classification (original)
  classify_player_states.py   # Player-level classification (new)
  classify_states_io.py       # Shared IO helpers, file paths, export functions
```

## Inputs

| Input | Source |
|-------|--------|
| Raw session CSVs | `datastore/raw/*_data.csv`, `*_chat.csv` |
| Promise classifications | `datastore/derived/promise_classifications.csv` |

## Outputs

| Output | Script |
|--------|--------|
| `datastore/derived/state_classification.csv` | `classify_states.py` |
| `datastore/derived/player_state_classification.csv` | `classify_player_states.py` |

## Key Results (10 sessions, 3,520 observations)

| Metric | Group-Level | Player-Level |
|--------|------------|-------------|
| Cooperative state | 2,528 (71.8%) | 2,309 (65.6%) |
| Noncooperative state | 992 (28.2%) | 1,211 (34.4%) |

The player-level approach reclassifies 219 observations from cooperative to noncooperative, capturing cases where a high contributor's own contribution inflated the group mean.

## Tests

- `tests/test_classify_states.py` — 26 unit tests for group-level
- `tests/test_classify_states_export.py` — 12 export tests for group-level
- `tests/test_classify_player_states.py` — 20 unit tests for player-level
- `tests/test_classify_player_states_export.py` — 12 export tests for player-level
- `tests/test_classify_states_integration.py` — integration tests
- Shared test helpers in `tests/conftest.py`
