# Issue #13: Verify contributions.csv matches raw data and experiment_data object

## Problem
The `contributions.csv` file is a critical derived dataset used throughout the analysis pipeline, but there was no automated verification that it accurately reflects the underlying oTree raw data and the experiment_data object structure.

## Solution
Created comprehensive test suite to verify data integrity:

### Test Coverage
1. **Schema validation** (9 tests): Verify column structure, data types, value ranges, and uniqueness constraints
2. **Coverage validation** (5 tests): Ensure all sessions, treatments, rounds, and players are represented
3. **Raw data matching** (3 tests): Compare all 3,520 contribution rows against source oTree CSVs
4. **Experiment object matching** (2 tests): Verify consistency with experiment_data hierarchical structure
5. **Integration test** (1 test): Round-trip consistency check

### Implementation
- Created `analysis/tests/test_contributions_csv.py` (902 lines, 20 tests)
- Added `slow` pytest marker to `pyproject.toml` for long-running tests
- All 20 tests passing

## Inputs
- `analysis/datastore/derived/contributions.csv` (file under test)
- `analysis/datastore/raw/*_data.csv` (10 raw oTree session files)
- `analysis/experiment_data.py` (hierarchical data object)

## Outputs
- `analysis/tests/test_contributions_csv.py` (test suite)

## Impact
- Provides confidence in data pipeline integrity
- Catches potential data extraction bugs early
- Documents expected data structure and relationships
- Enables safe refactoring of data processing code
