# Test Suite Documentation

This document provides detailed documentation of the `experiment_data.py` test suite, explaining what each test verifies and how it determines pass/fail status.

## Overview

The test suite validates that `experiment_data.py` correctly loads and represents raw oTree experimental data. It consists of **53 tests** across **9 test modules** (plus 1 shared fixtures file) covering:

- Data integrity (contributions, payoffs, chat messages)
- Structure validation (groups, segments, rounds, participants)
- Calculations (group totals, individual shares, payoff formula)
- Cross-session testing (both T1 and T2 treatments)
- Integration testing (full pipeline verification)

### Running the Tests

```bash
# Run all tests
uv run pytest analysis/tests/ -v

# Run specific test file
uv run pytest analysis/tests/test_contributions.py -v

# Run integration tests only
uv run pytest -m integration -v

# Run with coverage
uv run pytest analysis/tests/ --cov=analysis
```

---

## Test Modules

### 1. conftest.py - Shared Test Fixtures

This module provides pytest fixtures that supply test data to all test modules.

#### Path Fixtures

| Fixture | Returns | Description |
|---------|---------|-------------|
| `raw_data_dir` | `Path` | Path to `analysis/datastore/raw/` |
| `t1_session_paths` | `tuple[Path, Path]` | Tuple of (data_csv_path, chat_csv_path) for session 01_t1 |
| `t2_session_paths` | `tuple[Path, Path]` | Tuple of paths for session 03_t2 |

#### DataFrame Fixtures

| Fixture | Returns | Description |
|---------|---------|-------------|
| `t1_raw_df` | `pd.DataFrame` | Raw T1 game data loaded from CSV |
| `t2_raw_df` | `pd.DataFrame` | Raw T2 game data loaded from CSV |
| `t1_chat_df` | `pd.DataFrame` | Raw T1 chat messages loaded from CSV |
| `t2_chat_df` | `pd.DataFrame` | Raw T2 chat messages loaded from CSV |

#### Session/Experiment Fixtures

| Fixture | Returns | Description |
|---------|---------|-------------|
| `loaded_t1_session` | `Session` | T1 session loaded via `load_experiment_data()` |
| `loaded_t2_session` | `Session` | T2 session loaded via `load_experiment_data()` |
| `sample_experiment` | `Experiment` | Both T1 and T2 sessions in one Experiment object |

All fixtures handle missing files gracefully with `pytest.skip()`.

---

### 2. test_contributions.py - Contribution Accuracy Tests

Verifies that `Player.contribution` values match raw CSV data and fall within valid ranges.

#### test_player_contribution_matches_raw_csv

**Purpose:** Verify every player's contribution in the loaded Session matches the raw CSV.

**How it works:**
1. Iterates through all supergames (1-5) and their rounds
2. For each player, extracts the contribution from raw CSV column `supergame{N}.{R}.player.contribution`
3. Compares to `Player.contribution` in the loaded Session object
4. Collects all mismatches and reports the first 10

**Pass condition:** All loaded contributions exactly match raw CSV values.

#### test_all_contributions_non_negative

**Purpose:** Ensure no negative contributions exist in the data.

**How it works:**
1. Iterates all supergames, rounds, and players
2. Checks if `player.contribution < 0`
3. Collects any negative values found

**Pass condition:** Zero negative contributions found.

#### test_contribution_values_in_valid_range

**Purpose:** Verify all contributions are between 0 and the endowment (25 points).

**How it works:**
1. Iterates all players across all supergames/rounds
2. Checks if contribution is outside range [0, 25]
3. Lists any out-of-range values

**Pass condition:** All contributions are within [0, 25].

#### test_contribution_sample_t2

**Purpose:** Same verification as `test_player_contribution_matches_raw_csv` but for T2 session.

**Pass condition:** T2 contributions match raw CSV values.

---

### 3. test_chat.py - Chat Message Accuracy Tests

Verifies that `ChatMessage` objects correctly represent raw chat CSV data.

#### Constants

```python
# Known session code mismatches verified to be same session via timestamp analysis
VERIFIED_SESSION_CODE_EXCEPTIONS = {
    ('irrzlgk2', 'z8dowljr'),  # 03_t2: verified same session via timeline analysis
}
```

#### test_chat_message_body_matches_raw

**Purpose:** Verify `ChatMessage.body` matches raw CSV body column.

**How it works:**
1. Extracts all unique message bodies from raw CSV
2. Extracts all unique bodies from loaded ChatMessage objects
3. Verifies loaded bodies exist in raw data
4. Spot-checks first 5 raw bodies can be found in loaded data

**Pass condition:** All message bodies match between loaded and raw data.

#### test_chat_message_timestamp_matches_raw

**Purpose:** Verify `ChatMessage.timestamp` matches raw CSV timestamp.

**How it works:**
1. Creates lookup: `body → [list of timestamps]` from raw CSV (handles duplicate messages)
2. For each loaded ChatMessage, verifies its timestamp is in the expected list
3. Uses exact floating-point comparison

**Pass condition:** All timestamps match raw CSV values.

#### test_chat_message_nickname_matches_raw

**Purpose:** Verify `ChatMessage.nickname` matches raw CSV nickname column.

**How it works:**
1. Creates lookup: `(body, timestamp) → nickname` from raw CSV
2. For each loaded ChatMessage, verifies nickname matches expected

**Pass condition:** All nicknames match raw CSV values.

#### test_chat_message_count_matches_raw

**Purpose:** Verify total message count equals raw CSV row count.

**How it works:**
1. Counts all ChatMessage objects in loaded Session
2. Counts rows in raw chat CSV
3. Compares counts

**Pass condition:** `len(loaded_messages) == len(raw_csv_rows)`

#### test_chat_sample_t2

**Purpose:** Same chat verifications for T2 session.

**How it works:**
1. Checks if session codes match between chat CSV and data CSV
2. If mismatch, checks against `VERIFIED_SESSION_CODE_EXCEPTIONS`
3. If exception exists, proceeds with verification; otherwise skips
4. Verifies message count, body, timestamp, and nickname

**Pass condition:** T2 chat data matches raw CSV (or skips if unverified session mismatch).

---

### 4. test_payoffs.py - Payoff Accuracy Tests

Verifies that `Player.payoff` values match raw CSV and follow the public goods payoff formula.

#### Payoff Formula
```
payoff = (endowment - contribution) + individual_share
       = (25 - contribution) + (group_total * 0.4)
```

#### test_player_payoff_matches_raw_csv

**Purpose:** Verify `Player.payoff` equals raw CSV value.

**How it works:**
1. Builds lookup: `participant.label → payoff` from raw CSV column `supergame1.1.player.payoff`
2. For each player in loaded Session, compares payoff using `pytest.approx()` (floating-point tolerance)

**Pass condition:** All payoffs match within floating-point tolerance.

#### test_payoff_calculation_formula

**Purpose:** Verify payoff follows formula: `(25 - contribution) + individual_share`.

**How it works:**
1. For each player, gets contribution and individual_share from raw CSV
2. Calculates expected payoff using formula
3. Compares to loaded payoff using `pytest.approx()`

**Pass condition:** All payoffs match formula calculation.

#### test_individual_share_matches_raw

**Purpose:** Verify `individual_share` values exist and are valid in raw CSV.

**How it works:**
1. Builds lookup of individual_share values from raw CSV
2. Verifies each player has a value and it's positive

**Pass condition:** All individual_share values present and positive.

#### test_payoff_sample_t2

**Purpose:** Same payoff verification for T2 session.

**Pass condition:** T2 payoffs match raw CSV values and formula.

---

### 5. test_groups.py - Group Formation Tests

Verifies correct group structure: 4 players per group, 4 groups per round.

#### test_four_players_per_group

**Purpose:** Every Group has exactly 4 players.

**How it works:**
1. Iterates all supergames → rounds → groups
2. Checks `len(group.players) == 4`

**Pass condition:** All groups have exactly 4 players.

#### test_four_groups_per_round

**Purpose:** Every Round has exactly 4 groups.

**How it works:**
1. Iterates all supergames → rounds
2. Checks `len(round_obj.groups) == 4`

**Pass condition:** All rounds have exactly 4 groups.

#### test_sixteen_players_per_round

**Purpose:** Every Round has 16 players total (4 groups × 4 players).

**How it works:**
1. Iterates all supergames → rounds
2. Checks `len(round_obj.players) == 16`

**Pass condition:** All rounds have exactly 16 players.

#### test_player_labels_unique_in_group

**Purpose:** No duplicate player labels within same group.

**How it works:**
1. For each group, collects all player labels
2. Compares list length to set length (detects duplicates)

**Pass condition:** No duplicate labels in any group.

#### test_player_group_id_matches_group

**Purpose:** `Player.group_id` matches parent `Group.group_id`.

**How it works:**
1. Iterates all groups and their players
2. Verifies `player.group_id == group.group_id`

**Pass condition:** All player group_id values match their parent group.

#### test_group_id_in_subsession_matches_raw

**Purpose:** `Group.group_id` matches raw CSV `group.id_in_subsession` column.

**How it works:**
1. Extracts unique group IDs from raw CSV
2. Compares to group IDs in loaded Session

**Pass condition:** Group ID sets match.

#### test_group_formation_t2

**Purpose:** T2 has same group structure.

**Pass condition:** All group structure tests pass for T2.

---

### 6. test_group_contributions.py - Group Contribution Calculations

Verifies group totals and individual share calculations.

#### Constants
```python
MPCR = 0.4  # Marginal Per Capita Return (multiplier)
```

#### test_group_total_contribution_equals_sum

**Purpose:** Raw CSV `total_contribution` equals sum of player contributions.

**How it works:**
1. For each group in raw CSV, sums player contributions
2. Compares sum to `group.total_contribution` column value

**Pass condition:** All group totals equal player sums.

#### test_group_total_contribution_matches_raw

**Purpose:** Loaded `Group.total_contribution` matches raw CSV value.

**How it works:**
1. For each loaded group, looks up raw CSV value
2. Compares using `pytest.approx()`

**Pass condition:** All loaded group totals match raw CSV.

#### test_loaded_player_sum_equals_group_total

**Purpose:** Sum of loaded player contributions equals loaded group total.

**How it works:**
1. For each group, sums `player.contribution` for all players
2. Compares to `group.total_contribution`

**Pass condition:** Internal consistency of loaded data.

#### test_individual_share_calculation

**Purpose:** Verify `individual_share = round(total_contribution * MPCR)`.

**How it works:**
1. For each group in raw CSV, calculates expected share
2. Compares to actual share value
3. Note: oTree rounds the result for display/payment

**Pass condition:** All individual shares match formula with rounding.

#### test_group_contribution_sample_t2

**Purpose:** T2 raw CSV group contributions are valid.

**Pass condition:** T2 group totals and shares are correctly calculated.

#### test_t2_loaded_matches_raw

**Purpose:** T2 loaded group data matches raw CSV.

**Pass condition:** T2 loaded group totals match raw CSV values.

---

### 7. test_segments.py - Segment Structure Tests

Verifies sessions contain expected segments.

#### Constants
```python
EXPECTED_SEGMENTS = [
    'introduction', 'supergame1', 'supergame2', 'supergame3',
    'supergame4', 'supergame5', 'finalresults'
]
SUPERGAME_NAMES = ['supergame1', 'supergame2', 'supergame3', 'supergame4', 'supergame5']
```

#### test_all_segments_present

**Purpose:** Session has all 7 expected segments.

**How it works:**
1. Gets set of segment names from loaded Session
2. Verifies each expected segment is present

**Pass condition:** All expected segments found.

#### test_segment_count

**Purpose:** Session has exactly 7 segments total.

**How it works:**
1. Counts segments in Session
2. Compares to expected count

**Pass condition:** `len(session.segments) == 7`

#### test_supergame_segment_names

**Purpose:** Supergame names are exactly 'supergame1' through 'supergame5'.

**How it works:**
1. Filters segments starting with 'supergame'
2. Sorts and compares to expected list

**Pass condition:** Supergame names match exactly.

#### test_get_supergame_returns_correct_segment

**Purpose:** `session.get_supergame(N)` returns segment named 'supergameN'.

**How it works:**
1. Calls `get_supergame()` for each N in 1-5
2. Verifies returned segment has correct name

**Pass condition:** All get_supergame() calls return correctly named segments.

#### test_introduction_segment_exists

**Purpose:** 'introduction' segment exists and is accessible.

**Pass condition:** `session.get_segment('introduction')` returns valid segment.

#### test_finalresults_segment_exists

**Purpose:** 'finalresults' segment exists and is accessible.

**Pass condition:** `session.get_segment('finalresults')` returns valid segment.

#### test_segments_t2

**Purpose:** T2 has identical segment structure.

**Pass condition:** All segment tests pass for T2.

---

### 8. test_rounds.py - Round Structure Tests

Verifies correct number of rounds per supergame.

#### Constants
```python
EXPECTED_ROUNDS_PER_SUPERGAME = {1: 3, 2: 4, 3: 3, 4: 7, 5: 5}
EXPECTED_TOTAL_ROUNDS = 22
```

#### test_supergame1_has_3_rounds through test_supergame5_has_5_rounds

**Purpose:** Each supergame has correct round count (3, 4, 3, 7, 5).

**How it works:**
1. Gets round count for specific supergame
2. Compares to expected value

**Pass condition:** Round count matches expected.

#### test_round_numbers_sequential

**Purpose:** Round numbers are 1, 2, 3, ... with no gaps.

**How it works:**
1. For each supergame, gets sorted list of round numbers
2. Verifies equals `[1, 2, ..., N]`

**Pass condition:** No gaps in round numbering.

#### test_total_supergame_rounds

**Purpose:** Total of 22 rounds across all supergames.

**How it works:**
1. Sums round counts for all 5 supergames
2. Compares to 22

**Pass condition:** Total equals 22.

#### test_round_structure_t2

**Purpose:** T2 has identical round structure.

**Pass condition:** All round tests pass for T2.

---

### 9. test_participants.py - Participant Label Mapping Tests

Verifies 16 participants with correct labels (A-R, skipping I and O).

#### Constants
```python
EXPECTED_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                   'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R']
```

#### test_participant_labels_match_raw

**Purpose:** `Session.participant_labels` mapping matches raw CSV.

**How it works:**
1. Extracts `(participant.id_in_session, participant.label)` pairs from raw CSV
2. Verifies each ID maps to correct label in loaded Session

**Pass condition:** All ID → label mappings match.

#### test_participant_id_to_label_mapping

**Purpose:** Complete mapping dictionary matches raw CSV.

**How it works:**
1. Builds expected mapping from raw CSV
2. Compares entire dict to `session.participant_labels`

**Pass condition:** Mapping dictionaries are identical.

#### test_player_label_consistent_across_rounds

**Purpose:** Same participant has same label throughout all rounds.

**How it works:**
1. Tracks `participant_id → label` across all segments and rounds
2. Verifies consistency

**Pass condition:** No label changes for any participant.

#### test_sixteen_participants

**Purpose:** Exactly 16 participants in session.

**Pass condition:** `len(session.participant_labels) == 16`

#### test_unique_participant_labels

**Purpose:** All labels are unique and match expected set.

**How it works:**
1. Checks for duplicate labels (list length vs set length)
2. Verifies label set equals EXPECTED_LABELS

**Pass condition:** 16 unique labels matching A-R (no I, O).

#### test_participants_t2

**Purpose:** T2 has same participant structure.

**Pass condition:** All participant tests pass for T2.

---

### 10. test_integration.py - End-to-End Integration Tests

Verifies complete data loading pipeline.

All tests in this module are marked with `@pytest.mark.integration`.

#### test_full_session_load_t1

**Purpose:** Complete T1 session loads without errors.

**How it works:**
1. Loads T1 session using `load_experiment_data()`
2. Verifies experiment has 1 session with treatment=1
3. Checks all 5 supergames exist with rounds

**Pass condition:** Full session structure loads correctly.

#### test_full_session_load_t2

**Purpose:** Complete T2 session loads without errors.

**Pass condition:** T2 loads correctly with treatment=2.

#### test_experiment_with_both_sessions

**Purpose:** Both T1 and T2 can be loaded into single Experiment.

**How it works:**
1. Uses `sample_experiment` fixture (both sessions)
2. Verifies 2 sessions present
3. Checks both treatment values (1 and 2) exist

**Pass condition:** Multi-session Experiment loads correctly.

#### test_to_dataframe_contributions_has_expected_columns

**Purpose:** DataFrame from `to_dataframe_contributions()` has correct columns.

**Expected columns:**
```python
['session_code', 'treatment', 'segment', 'round', 'group',
 'label', 'participant_id', 'contribution', 'payoff', 'role']
```

**How it works:**
1. Gets DataFrame from experiment
2. Checks for missing columns
3. Checks for unexpected extra columns

**Pass condition:** All expected columns present, no extras.

#### test_to_dataframe_contributions_row_count

**Purpose:** DataFrame row count matches expected from raw data.

**How it works:**
1. Counts valid contribution rows in T1 and T2 raw CSVs
2. Sums expected counts
3. Compares to actual DataFrame length

**Pass condition:** Row counts match within tolerance.

#### test_random_sample_verification

**Purpose:** Random sample of data points match raw CSV values.

**How it works:**
1. Seeds random generator (42) for reproducibility
2. Samples up to 10 participants with random supergame/round combinations
3. Looks up values in both raw CSV and loaded DataFrame
4. Compares values using `pytest.approx()`

**Pass condition:** All sampled values match raw data.

---

## Test Results Summary

When all tests pass:
```
======================== 53 passed in 2.12s ========================
```

### Bugs Found and Fixed

During test development, the following bugs were discovered and fixed in `experiment_data.py`:

1. **Line 773-774:** Used `player.id_in_group` instead of `group.id_in_subsession` when loading group-level data, causing all groups to receive group 1's data.

2. **Line 542:** Channel pattern regex was hardcoded to `^1\-supergame...` but some sessions use different prefixes (e.g., `5-supergame...`). Fixed to `^\d+\-supergame...`.

### Known Data Issues Handled

1. **Session code mismatch (03_t2):** Chat and data CSVs have different session codes (`z8dowljr` vs `irrzlgk2`) but are verified to be the same session via timestamp alignment. Added to `VERIFIED_SESSION_CODE_EXCEPTIONS`.

2. **Floating-point comparisons:** All monetary values use `pytest.approx()` for tolerance.

3. **Missing data files:** Fixtures use `pytest.skip()` when raw data files are unavailable.

---

## Adding New Tests

When adding tests, follow these patterns:

1. **Use fixtures** from `conftest.py` for data access
2. **Test both T1 and T2** sessions when applicable
3. **Use descriptive assertions** that show expected vs actual values
4. **Mark integration tests** with `@pytest.mark.integration`
5. **Use `pytest.approx()`** for floating-point comparisons
