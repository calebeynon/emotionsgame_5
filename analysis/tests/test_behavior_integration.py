"""
Integration tests for behavior classification output CSV.

Tests validate the behavior_classifications.csv output file for correctness,
including column structure, row counts, and classification logic constraints.

Author: Claude Code
Date: 2026-01-17
"""

import pytest
import pandas as pd
from pathlib import Path

# FILE PATHS
DATA_DIR = Path(__file__).parent.parent / 'datastore'
OUTPUT_FILE = DATA_DIR / 'derived' / 'behavior_classifications.csv'

# EXPECTED VALUES
EXPECTED_COLUMNS = [
    'session_code', 'treatment', 'segment', 'round', 'group', 'label',
    'participant_id', 'contribution', 'payoff', 'made_promise',
    'is_liar_strict', 'is_liar_lenient', 'is_sucker_strict', 'is_sucker_lenient'
]
EXPECTED_ROW_COUNT = 3520  # 10 sessions * 16 players * 22 rounds
FLAG_COLUMNS = ['is_liar_strict', 'is_liar_lenient', 'is_sucker_strict', 'is_sucker_lenient']


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def behavior_df():
    """Load behavior classifications CSV once for all tests."""
    return pd.read_csv(OUTPUT_FILE)


# =====
# Test output file existence and structure
# =====
@pytest.mark.integration
class TestOutputFileStructure:
    """Tests for output file existence and column structure."""

    def test_output_csv_exists(self):
        """Verify the output CSV file is created after running the script."""
        assert OUTPUT_FILE.exists(), f"Output file not found at {OUTPUT_FILE}"

    def test_output_csv_columns(self, behavior_df):
        """Verify output has expected columns."""
        actual_columns = list(behavior_df.columns)
        assert actual_columns == EXPECTED_COLUMNS, (
            f"Column mismatch.\nExpected: {EXPECTED_COLUMNS}\nActual: {actual_columns}"
        )

    def test_output_row_count(self, behavior_df):
        """Verify all player-rounds are included (3520 = 10 sessions * 16 players * 22 rounds)."""
        assert len(behavior_df) == EXPECTED_ROW_COUNT, (
            f"Expected {EXPECTED_ROW_COUNT} rows, got {len(behavior_df)}"
        )


# =====
# Test round 1 behavior
# =====
@pytest.mark.integration
class TestRound1Flags:
    """Tests that round 1 always has False for all behavior flags."""

    def test_no_flags_in_round_1(self, behavior_df):
        """Verify all round 1 records have all four flags = False."""
        round_1 = behavior_df[behavior_df['round'] == 1]

        for flag_col in FLAG_COLUMNS:
            flagged_count = round_1[flag_col].sum()
            assert flagged_count == 0, (
                f"Round 1 should have no {flag_col}=True, but found {flagged_count}"
            )


# =====
# Test flag persistence within segment
# =====
@pytest.mark.integration
class TestFlagPersistence:
    """Tests that flags persist correctly within segments."""

    def test_flag_persistence_within_segment(self, behavior_df):
        """Verify once a flag is True, it stays True for all subsequent rounds in that segment."""
        for flag_col in FLAG_COLUMNS:
            violations = find_persistence_violations(behavior_df, flag_col)
            assert len(violations) == 0, (
                f"Flag {flag_col} has persistence violations:\n{violations.head(10)}"
            )


def find_persistence_violations(df: pd.DataFrame, flag_col: str) -> pd.DataFrame:
    """Find cases where flag is True then False in same segment for same player."""
    violations = []

    for (session, segment, label), group in df.groupby(['session_code', 'segment', 'label']):
        group = group.sort_values('round')
        flag_values = group[flag_col].tolist()
        rounds = group['round'].tolist()

        flag_became_true = False
        for i, (rnd, flag_val) in enumerate(zip(rounds, flag_values)):
            if flag_val:
                flag_became_true = True
            elif flag_became_true and not flag_val:
                violations.append({
                    'session_code': session,
                    'segment': segment,
                    'label': label,
                    'round': rnd,
                    'flag': flag_col,
                    'issue': 'Flag went from True to False'
                })

    return pd.DataFrame(violations)


# =====
# Test flag reset across segments
# =====
@pytest.mark.integration
class TestFlagReset:
    """Tests that flags reset between segments."""

    def test_flag_reset_across_segments(self, behavior_df):
        """Verify flags reset to False at the start of each new segment (supergame)."""
        round_1_of_each_segment = behavior_df[behavior_df['round'] == 1]

        for flag_col in FLAG_COLUMNS:
            flagged_in_round_1 = round_1_of_each_segment[round_1_of_each_segment[flag_col] == True]
            assert len(flagged_in_round_1) == 0, (
                f"Round 1 of segment should have {flag_col}=False, but found:\n"
                f"{flagged_in_round_1[['session_code', 'segment', 'label', flag_col]].head()}"
            )


# =====
# Test liar requires promise
# =====
@pytest.mark.integration
class TestLiarRequiresPromise:
    """Tests that liar flags require prior promises."""

    def test_liar_requires_promise(self, behavior_df):
        """Verify no player has is_liar_* = True without having made_promise = True in prior round of same segment."""
        for threshold in ['strict', 'lenient']:
            flag_col = f'is_liar_{threshold}'
            violations = find_liar_without_promise(behavior_df, flag_col)
            assert len(violations) == 0, (
                f"Found {flag_col}=True without prior made_promise=True:\n{violations.head()}"
            )


def find_liar_without_promise(df: pd.DataFrame, flag_col: str) -> pd.DataFrame:
    """Find liars who never made a promise in prior rounds of same segment."""
    violations = []

    for (session, segment, label), group in df.groupby(['session_code', 'segment', 'label']):
        group = group.sort_values('round')
        made_promise_prior = False

        for _, row in group.iterrows():
            if row[flag_col] and not made_promise_prior:
                violations.append({
                    'session_code': session,
                    'segment': segment,
                    'label': label,
                    'round': row['round'],
                    'flag': flag_col,
                    'issue': 'Liar flag without prior promise'
                })
            if row['made_promise']:
                made_promise_prior = True

    return pd.DataFrame(violations)


# =====
# Test sucker requires 25 contribution
# =====
@pytest.mark.integration
class TestSuckerRequires25:
    """Tests that sucker flags require prior 25 contribution."""

    def test_sucker_requires_25_contribution(self, behavior_df):
        """Verify no player has is_sucker_* = True without having contributed 25 in prior round of same segment."""
        for threshold in ['strict', 'lenient']:
            flag_col = f'is_sucker_{threshold}'
            violations = find_sucker_without_25(behavior_df, flag_col)
            assert len(violations) == 0, (
                f"Found {flag_col}=True without prior contribution=25:\n{violations.head()}"
            )


def find_sucker_without_25(df: pd.DataFrame, flag_col: str) -> pd.DataFrame:
    """Find suckers who never contributed 25 in prior rounds of same segment."""
    violations = []

    for (session, segment, label), group in df.groupby(['session_code', 'segment', 'label']):
        group = group.sort_values('round')
        contributed_25_prior = False

        for _, row in group.iterrows():
            if row[flag_col] and not contributed_25_prior:
                violations.append({
                    'session_code': session,
                    'segment': segment,
                    'label': label,
                    'round': row['round'],
                    'flag': flag_col,
                    'contribution_history': group[group['round'] < row['round']]['contribution'].tolist(),
                    'issue': 'Sucker flag without prior 25 contribution'
                })
            if row['contribution'] == 25:
                contributed_25_prior = True

    return pd.DataFrame(violations)
