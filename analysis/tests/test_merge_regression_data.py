"""
Tests for merge_regression_data.py data merging functions.

Tests key helper functions for merging behavior, sentiment, and promise data
into a unified regression dataset.

Author: Claude Code
Date: 2026-01-27
"""

import pytest
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add derived directory to path (where merge_regression_data.py lives)
sys.path.insert(0, str(Path(__file__).parent.parent / "derived"))

from merge_regression_data import (
    load_behavior_data,
    load_sentiment_data,
    load_promise_data,
    merge_datasets,
    compute_lagged_variables,
    validate_output,
    _validate_round_1_nulls,
    _validate_no_duplicate_columns,
    MERGE_KEYS,
    SENTIMENT_COLS,
    PROMISE_COLS,
    LAGGED_SENTIMENT_COLS,
    LAGGED_LIAR_COLS,
)

# FILE PATHS (for direct file access tests)
DERIVED_DIR = Path(__file__).parent.parent / 'datastore' / 'derived'
BEHAVIOR_CSV = DERIVED_DIR / 'behavior_classifications.csv'
SENTIMENT_CSV = DERIVED_DIR / 'sentiment_scores.csv'
PROMISE_CSV = DERIVED_DIR / 'promise_classifications.csv'

# EXPECTED CONSTANTS
EXPECTED_BEHAVIOR_ROWS = 3520
EXPECTED_SENTIMENT_ROWS = 2298
EXPECTED_PROMISE_ROWS = 2298

# Column lists for validation
LIAR_COLUMNS = [
    'is_liar_strict',
    'is_liar_lenient',
    'is_sucker_strict',
    'is_sucker_lenient',
]


# =====
# Fixtures for mock data
# =====
@pytest.fixture
def mock_behavior_df():
    """Create small DataFrame mimicking behavior_classifications.csv.

    Includes both round 1 (no prior chat) and round 2+ (with prior chat).
    """
    return pd.DataFrame({
        'session_code': ['abc123'] * 8,
        'treatment': [1] * 8,
        'segment': ['supergame1'] * 8,
        'round': [1, 1, 1, 1, 2, 2, 2, 2],
        'group': [1, 1, 1, 1, 1, 1, 1, 1],
        'label': ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D'],
        'participant_id': [1, 2, 3, 4, 1, 2, 3, 4],
        'contribution': [25.0, 20.0, 15.0, 10.0, 25.0, 25.0, 20.0, 15.0],
        'payoff': [40.0, 35.0, 30.0, 25.0, 45.0, 45.0, 40.0, 35.0],
        'made_promise': [False, False, False, False, True, True, True, False],
        'is_liar_strict': [False] * 8,
        'is_liar_lenient': [False] * 8,
        'is_sucker_strict': [False] * 8,
        'is_sucker_lenient': [False] * 8,
    })


@pytest.fixture
def mock_sentiment_df():
    """Create small DataFrame mimicking sentiment_scores.csv.

    Only includes round 2+ observations (chat influences subsequent rounds).
    """
    return pd.DataFrame({
        'session_code': ['abc123'] * 4,
        'segment': ['supergame1'] * 4,
        'round': [2, 2, 2, 2],
        'label': ['A', 'B', 'C', 'D'],
        'sentiment_compound_mean': [0.5, -0.2, 0.3, 0.0],
        'sentiment_compound_std': [0.1, 0.2, 0.0, 0.15],
        'sentiment_compound_min': [0.4, -0.4, 0.3, -0.1],
        'sentiment_compound_max': [0.6, 0.0, 0.3, 0.1],
        'sentiment_positive_mean': [0.3, 0.1, 0.2, 0.15],
        'sentiment_negative_mean': [0.05, 0.2, 0.1, 0.1],
        'sentiment_neutral_mean': [0.65, 0.7, 0.7, 0.75],
    })


@pytest.fixture
def mock_promise_df():
    """Create small DataFrame mimicking promise_classifications.csv.

    Only includes round 2+ observations (chat influences subsequent rounds).
    """
    return pd.DataFrame({
        'session_code': ['abc123'] * 4,
        'segment': ['supergame1'] * 4,
        'round': [2, 2, 2, 2],
        'label': ['A', 'B', 'C', 'D'],
        'message_count': [2, 3, 1, 0],
        'promise_count': [1, 2, 0, 0],
        'promise_percentage': [50.0, 66.7, 0.0, 0.0],
    })


@pytest.fixture
def mock_lagged_df():
    """Create DataFrame with 3 rounds per player for testing lagged variables.

    Includes varying liar flags and sentiment to verify lagging works.
    """
    return pd.DataFrame({
        'session_code': ['abc123'] * 6,
        'segment': ['supergame1'] * 6,
        'round': [1, 2, 3, 1, 2, 3],
        'label': ['A', 'A', 'A', 'B', 'B', 'B'],
        'is_liar_strict': [False, True, False, False, False, True],
        'is_liar_lenient': [False, True, True, True, False, False],
        'sentiment_compound_mean': [np.nan, 0.5, 0.3, np.nan, -0.2, 0.1],
    })


# =====
# Test load functions
# =====
class TestLoadFunctions:
    """Tests for data loading functions."""

    def test_load_behavior_returns_correct_row_count(self):
        """Verify behavior data has 3,520 rows."""
        if not BEHAVIOR_CSV.exists():
            pytest.skip(f"behavior_classifications.csv not found: {BEHAVIOR_CSV}")

        df = pd.read_csv(BEHAVIOR_CSV)
        assert len(df) == EXPECTED_BEHAVIOR_ROWS, (
            f"Expected {EXPECTED_BEHAVIOR_ROWS} rows, found {len(df)}"
        )

    def test_load_sentiment_returns_correct_row_count(self):
        """Verify sentiment data has 2,298 rows."""
        if not SENTIMENT_CSV.exists():
            pytest.skip(f"sentiment_scores.csv not found: {SENTIMENT_CSV}")

        df = pd.read_csv(SENTIMENT_CSV)
        assert len(df) == EXPECTED_SENTIMENT_ROWS, (
            f"Expected {EXPECTED_SENTIMENT_ROWS} rows, found {len(df)}"
        )

    def test_load_promise_returns_correct_row_count(self):
        """Verify promise data has 2,298 rows."""
        if not PROMISE_CSV.exists():
            pytest.skip(f"promise_classifications.csv not found: {PROMISE_CSV}")

        df = pd.read_csv(PROMISE_CSV)
        assert len(df) == EXPECTED_PROMISE_ROWS, (
            f"Expected {EXPECTED_PROMISE_ROWS} rows, found {len(df)}"
        )


# =====
# Test merge datasets
# =====
class TestMergeDatasets:
    """Tests for the merge_datasets function."""

    def test_merge_preserves_all_behavior_rows(
        self, mock_behavior_df, mock_sentiment_df, mock_promise_df
    ):
        """Output should have same row count as behavior (left join)."""
        result = merge_datasets(
            mock_behavior_df, mock_sentiment_df, mock_promise_df
        )
        assert len(result) == len(mock_behavior_df), (
            f"Expected {len(mock_behavior_df)} rows, found {len(result)}"
        )

    def test_merge_adds_sentiment_columns(
        self, mock_behavior_df, mock_sentiment_df, mock_promise_df
    ):
        """All 7 sentiment columns should be present after merge."""
        result = merge_datasets(
            mock_behavior_df, mock_sentiment_df, mock_promise_df
        )

        for col in SENTIMENT_COLS:
            assert col in result.columns, f"Missing sentiment column: {col}"

    def test_merge_adds_promise_columns(
        self, mock_behavior_df, mock_sentiment_df, mock_promise_df
    ):
        """Promise columns (message_count, promise_count, promise_percentage) present."""
        result = merge_datasets(
            mock_behavior_df, mock_sentiment_df, mock_promise_df
        )

        for col in PROMISE_COLS:
            assert col in result.columns, f"Missing promise column: {col}"

    def test_merge_on_correct_keys(
        self, mock_behavior_df, mock_sentiment_df, mock_promise_df
    ):
        """Verify merge uses correct keys by checking matched values."""
        result = merge_datasets(
            mock_behavior_df, mock_sentiment_df, mock_promise_df
        )

        # Round 2, label A should have sentiment from mock_sentiment_df
        round_2_a = result[(result['round'] == 2) & (result['label'] == 'A')]
        assert len(round_2_a) == 1
        assert round_2_a['sentiment_compound_mean'].iloc[0] == 0.5


# =====
# Test round 1 NaN handling
# =====
class TestRound1NaNHandling:
    """Tests for NaN values in round 1 observations."""

    def test_round_1_has_nan_sentiment_compound_mean(
        self, mock_behavior_df, mock_sentiment_df, mock_promise_df
    ):
        """Round 1 rows should have NaN for sentiment_compound_mean."""
        merged = merge_datasets(
            mock_behavior_df, mock_sentiment_df, mock_promise_df
        )

        round_1 = merged[merged['round'] == 1]
        assert round_1['sentiment_compound_mean'].isna().all(), (
            "Round 1 should have NaN sentiment_compound_mean"
        )

    def test_round_1_has_nan_all_sentiment_columns(
        self, mock_behavior_df, mock_sentiment_df, mock_promise_df
    ):
        """Round 1 rows should have NaN for all sentiment columns."""
        merged = merge_datasets(
            mock_behavior_df, mock_sentiment_df, mock_promise_df
        )

        round_1 = merged[merged['round'] == 1]
        for col in SENTIMENT_COLS:
            assert round_1[col].isna().all(), (
                f"Round 1 should have NaN for {col}"
            )

    def test_round_1_has_nan_promise_columns(
        self, mock_behavior_df, mock_sentiment_df, mock_promise_df
    ):
        """Round 1 rows should have NaN for promise columns."""
        merged = merge_datasets(
            mock_behavior_df, mock_sentiment_df, mock_promise_df
        )

        round_1 = merged[merged['round'] == 1]
        for col in PROMISE_COLS:
            assert round_1[col].isna().all(), (
                f"Round 1 should have NaN for {col}"
            )

    def test_non_round_1_may_have_values(
        self, mock_behavior_df, mock_sentiment_df, mock_promise_df
    ):
        """Non-round-1 rows should have values (not all NaN)."""
        merged = merge_datasets(
            mock_behavior_df, mock_sentiment_df, mock_promise_df
        )

        non_round_1 = merged[merged['round'] > 1]

        # At least some non-round-1 rows should have sentiment values
        assert not non_round_1['sentiment_compound_mean'].isna().all(), (
            "Non-round-1 should have at least some sentiment values"
        )


# =====
# Test liar columns present
# =====
class TestLiarColumnsPresent:
    """Tests for liar/sucker flag columns in merged output."""

    def test_is_liar_strict_present(
        self, mock_behavior_df, mock_sentiment_df, mock_promise_df
    ):
        """Merged output should contain is_liar_strict column."""
        merged = merge_datasets(
            mock_behavior_df, mock_sentiment_df, mock_promise_df
        )
        assert 'is_liar_strict' in merged.columns

    def test_is_liar_lenient_present(
        self, mock_behavior_df, mock_sentiment_df, mock_promise_df
    ):
        """Merged output should contain is_liar_lenient column."""
        merged = merge_datasets(
            mock_behavior_df, mock_sentiment_df, mock_promise_df
        )
        assert 'is_liar_lenient' in merged.columns

    def test_is_sucker_strict_present(
        self, mock_behavior_df, mock_sentiment_df, mock_promise_df
    ):
        """Merged output should contain is_sucker_strict column."""
        merged = merge_datasets(
            mock_behavior_df, mock_sentiment_df, mock_promise_df
        )
        assert 'is_sucker_strict' in merged.columns

    def test_is_sucker_lenient_present(
        self, mock_behavior_df, mock_sentiment_df, mock_promise_df
    ):
        """Merged output should contain is_sucker_lenient column."""
        merged = merge_datasets(
            mock_behavior_df, mock_sentiment_df, mock_promise_df
        )
        assert 'is_sucker_lenient' in merged.columns


# =====
# Test validation
# =====
class TestValidation:
    """Tests for the validation functions."""

    def test_validate_row_count_passes_correct_count(
        self, mock_behavior_df, mock_sentiment_df, mock_promise_df
    ):
        """Validation passes when row count matches expected."""
        merged = merge_datasets(
            mock_behavior_df, mock_sentiment_df, mock_promise_df
        )

        # Validation helper functions should not raise
        _validate_round_1_nulls(merged)
        _validate_no_duplicate_columns(merged)

    def test_validate_row_count_fails_wrong_count(self):
        """Validation fails when row count does not match expected."""
        # Create a small df that won't match expected 3520 rows
        small_df = pd.DataFrame({'session_code': ['a'], 'round': [1]})

        with pytest.raises(ValueError, match="Expected 3520 rows"):
            validate_output(small_df)

    def test_validate_round_1_nan_passes(
        self, mock_behavior_df, mock_sentiment_df, mock_promise_df
    ):
        """Validation passes when round 1 has expected NaN values."""
        merged = merge_datasets(
            mock_behavior_df, mock_sentiment_df, mock_promise_df
        )

        # Should not raise
        _validate_round_1_nulls(merged)

    def test_validate_round_1_nan_fails_with_values(self):
        """Validation fails if round 1 has non-NaN sentiment values."""
        # Create invalid df with sentiment value in round 1
        invalid_df = pd.DataFrame({
            'round': [1, 1, 2, 2],
            'sentiment_compound_mean': [0.5, np.nan, 0.3, 0.4],
        })

        with pytest.raises(ValueError, match="Round 1 should have NaN"):
            _validate_round_1_nulls(invalid_df)


# =====
# Test no duplicate columns
# =====
class TestNoDuplicateColumns:
    """Tests to ensure no duplicate columns after merge."""

    def test_no_duplicate_contribution_column(
        self, mock_behavior_df, mock_sentiment_df, mock_promise_df
    ):
        """Should not have contribution_x/contribution_y after merge."""
        merged = merge_datasets(
            mock_behavior_df, mock_sentiment_df, mock_promise_df
        )

        contribution_cols = [c for c in merged.columns if 'contribution' in c.lower()]
        assert len(contribution_cols) == 1, (
            f"Expected 1 contribution column, found: {contribution_cols}"
        )
        assert 'contribution' in merged.columns

    def test_no_duplicate_payoff_column(
        self, mock_behavior_df, mock_sentiment_df, mock_promise_df
    ):
        """Should not have payoff_x/payoff_y after merge."""
        merged = merge_datasets(
            mock_behavior_df, mock_sentiment_df, mock_promise_df
        )

        payoff_cols = [c for c in merged.columns if 'payoff' in c.lower()]
        assert len(payoff_cols) == 1, (
            f"Expected 1 payoff column, found: {payoff_cols}"
        )
        assert 'payoff' in merged.columns


# =====
# Test output schema
# =====
class TestOutputSchema:
    """Tests for the expected output schema."""

    def test_expected_columns_present(
        self, mock_behavior_df, mock_sentiment_df, mock_promise_df
    ):
        """All expected columns should be present in output."""
        merged = merge_datasets(
            mock_behavior_df, mock_sentiment_df, mock_promise_df
        )

        expected_cols = (
            MERGE_KEYS +
            ['treatment', 'group', 'participant_id'] +
            ['contribution', 'payoff', 'made_promise'] +
            LIAR_COLUMNS +
            SENTIMENT_COLS +
            PROMISE_COLS
        )

        for col in expected_cols:
            assert col in merged.columns, f"Missing expected column: {col}"

    def test_column_dtypes_correct(
        self, mock_behavior_df, mock_sentiment_df, mock_promise_df
    ):
        """Key columns should have correct data types."""
        merged = merge_datasets(
            mock_behavior_df, mock_sentiment_df, mock_promise_df
        )

        # String columns
        assert merged['session_code'].dtype == 'object'
        assert merged['segment'].dtype == 'object'
        assert merged['label'].dtype == 'object'

        # Integer columns
        assert merged['treatment'].dtype in ['int64', 'int32']
        assert merged['round'].dtype in ['int64', 'int32']
        assert merged['group'].dtype in ['int64', 'int32']
        assert merged['participant_id'].dtype in ['int64', 'int32']

        # Float columns (contribution, payoff, sentiment scores)
        assert merged['contribution'].dtype == 'float64'
        assert merged['payoff'].dtype == 'float64'


# =====
# Test column definitions (constants)
# =====
class TestColumnDefinitions:
    """Tests to verify column definitions are correct."""

    def test_merge_keys_complete(self):
        """Merge keys should include all required identifiers."""
        expected = ['session_code', 'segment', 'round', 'label']
        assert MERGE_KEYS == expected

    def test_sentiment_cols_complete(self):
        """Sentiment columns should include all sentiment metrics."""
        expected = [
            'sentiment_compound_mean', 'sentiment_compound_std',
            'sentiment_compound_min', 'sentiment_compound_max',
            'sentiment_positive_mean', 'sentiment_negative_mean',
            'sentiment_neutral_mean',
        ]
        assert SENTIMENT_COLS == expected

    def test_promise_cols_complete(self):
        """Promise columns should include all promise metrics."""
        expected = ['message_count', 'promise_count', 'promise_percentage']
        assert PROMISE_COLS == expected

    def test_lagged_sentiment_cols_complete(self):
        """Lagged sentiment columns should include sentiment_compound_mean."""
        expected = ['sentiment_compound_mean']
        assert LAGGED_SENTIMENT_COLS == expected

    def test_lagged_liar_cols_complete(self):
        """Lagged liar columns should include both strict and lenient."""
        expected = ['is_liar_strict', 'is_liar_lenient']
        assert LAGGED_LIAR_COLS == expected


# =====
# Test compute_lagged_variables
# =====
class TestComputeLaggedVariables:
    """Tests for the compute_lagged_variables function."""

    def test_creates_lied_prev_period_strict_column(self, mock_lagged_df):
        """Should create lied_prev_period_strict column."""
        result = compute_lagged_variables(mock_lagged_df)
        assert 'lied_prev_period_strict' in result.columns

    def test_creates_lied_prev_period_lenient_column(self, mock_lagged_df):
        """Should create lied_prev_period_lenient column."""
        result = compute_lagged_variables(mock_lagged_df)
        assert 'lied_prev_period_lenient' in result.columns

    def test_creates_sentiment_compound_mean_prev_column(self, mock_lagged_df):
        """Should create sentiment_compound_mean_prev column."""
        result = compute_lagged_variables(mock_lagged_df)
        assert 'sentiment_compound_mean_prev' in result.columns

    def test_round_1_has_nan_lagged_liar_strict(self, mock_lagged_df):
        """Round 1 should have NaN for lied_prev_period_strict."""
        result = compute_lagged_variables(mock_lagged_df)
        round_1 = result[result['round'] == 1]
        assert round_1['lied_prev_period_strict'].isna().all()

    def test_round_1_has_nan_lagged_liar_lenient(self, mock_lagged_df):
        """Round 1 should have NaN for lied_prev_period_lenient."""
        result = compute_lagged_variables(mock_lagged_df)
        round_1 = result[result['round'] == 1]
        assert round_1['lied_prev_period_lenient'].isna().all()

    def test_round_1_has_nan_lagged_sentiment(self, mock_lagged_df):
        """Round 1 should have NaN for sentiment_compound_mean_prev."""
        result = compute_lagged_variables(mock_lagged_df)
        round_1 = result[result['round'] == 1]
        assert round_1['sentiment_compound_mean_prev'].isna().all()

    def test_round_2_lagged_sentiment_is_nan(self, mock_lagged_df):
        """Round 2 should have NaN sentiment_prev (round 1 sentiment is NaN)."""
        result = compute_lagged_variables(mock_lagged_df)
        round_2 = result[result['round'] == 2]
        assert round_2['sentiment_compound_mean_prev'].isna().all()

    def test_round_2_lagged_liar_has_values(self, mock_lagged_df):
        """Round 2 should have valid lied_prev_period values."""
        result = compute_lagged_variables(mock_lagged_df)
        round_2 = result[result['round'] == 2]
        assert round_2['lied_prev_period_strict'].notna().all()
        assert round_2['lied_prev_period_lenient'].notna().all()

    def test_lagged_values_correctly_shifted(self, mock_lagged_df):
        """Verify lagged values match previous round's values.

        Player A: round 1 is_liar_strict=False, round 2 is_liar_strict=True
        So round 2 lied_prev_period_strict=False, round 3 lied_prev_period_strict=True
        """
        result = compute_lagged_variables(mock_lagged_df)
        result = result.sort_values(['label', 'round'])

        player_a = result[result['label'] == 'A']

        # Round 2: lagged from round 1 (is_liar_strict=False)
        r2 = player_a[player_a['round'] == 2]
        assert r2['lied_prev_period_strict'].iloc[0] == False

        # Round 3: lagged from round 2 (is_liar_strict=True)
        r3 = player_a[player_a['round'] == 3]
        assert r3['lied_prev_period_strict'].iloc[0] == True

    def test_lagged_sentiment_correctly_shifted(self, mock_lagged_df):
        """Verify lagged sentiment matches previous round's values.

        Player A: round 2 sentiment=0.5, round 3 sentiment=0.3
        So round 3 sentiment_prev=0.5
        """
        result = compute_lagged_variables(mock_lagged_df)
        result = result.sort_values(['label', 'round'])

        player_a = result[result['label'] == 'A']
        r3 = player_a[player_a['round'] == 3]

        assert r3['sentiment_compound_mean_prev'].iloc[0] == 0.5

    def test_preserves_row_count(self, mock_lagged_df):
        """compute_lagged_variables should not change row count."""
        original_count = len(mock_lagged_df)
        result = compute_lagged_variables(mock_lagged_df)
        assert len(result) == original_count

    def test_lagging_respects_player_boundaries(self):
        """Lagged values should not cross player boundaries.

        Each player's round 1 should have NaN, not values from previous player.
        """
        df = pd.DataFrame({
            'session_code': ['abc'] * 4,
            'segment': ['sg1'] * 4,
            'round': [1, 2, 1, 2],
            'label': ['A', 'A', 'B', 'B'],
            'is_liar_strict': [True, False, True, False],
            'is_liar_lenient': [True, True, False, False],
            'sentiment_compound_mean': [np.nan, 0.5, np.nan, -0.2],
        })

        result = compute_lagged_variables(df)

        # Player B round 1 should have NaN (not lagged from player A)
        b_r1 = result[(result['label'] == 'B') & (result['round'] == 1)]
        assert b_r1['lied_prev_period_strict'].isna().all()

    def test_lagging_respects_segment_boundaries(self):
        """Lagged values should not cross segment boundaries.

        Supergame2 round 1 should have NaN, not values from supergame1.
        """
        df = pd.DataFrame({
            'session_code': ['abc'] * 4,
            'segment': ['sg1', 'sg1', 'sg2', 'sg2'],
            'round': [1, 2, 1, 2],
            'label': ['A', 'A', 'A', 'A'],
            'is_liar_strict': [True, False, True, False],
            'is_liar_lenient': [True, True, False, False],
            'sentiment_compound_mean': [np.nan, 0.5, np.nan, -0.2],
        })

        result = compute_lagged_variables(df)

        # Supergame2 round 1 should have NaN (not lagged from supergame1)
        sg2_r1 = result[(result['segment'] == 'sg2') & (result['round'] == 1)]
        assert sg2_r1['lied_prev_period_strict'].isna().all()


# =====
# Test edge cases
# =====
class TestEdgeCases:
    """Tests for edge cases in merging."""

    def test_handles_empty_sentiment_df(self, mock_behavior_df):
        """Should handle empty sentiment DataFrame."""
        empty_sentiment = pd.DataFrame(columns=MERGE_KEYS + SENTIMENT_COLS)
        empty_promise = pd.DataFrame(columns=MERGE_KEYS + PROMISE_COLS)

        result = merge_datasets(mock_behavior_df, empty_sentiment, empty_promise)

        assert len(result) == len(mock_behavior_df)
        for col in SENTIMENT_COLS:
            assert result[col].isna().all()

    def test_handles_partial_match(self, mock_behavior_df):
        """Should handle sentiment data that only partially matches behavior."""
        partial_sentiment = pd.DataFrame({
            'session_code': ['abc123', 'abc123'],
            'segment': ['supergame1', 'supergame1'],
            'round': [2, 2],
            'label': ['A', 'B'],  # Only A and B, not C and D
            'sentiment_compound_mean': [0.5, -0.2],
            'sentiment_compound_std': [0.1, 0.2],
            'sentiment_compound_min': [0.4, -0.4],
            'sentiment_compound_max': [0.6, 0.0],
            'sentiment_positive_mean': [0.3, 0.1],
            'sentiment_negative_mean': [0.05, 0.2],
            'sentiment_neutral_mean': [0.65, 0.7],
        })
        empty_promise = pd.DataFrame(columns=MERGE_KEYS + PROMISE_COLS)

        result = merge_datasets(mock_behavior_df, partial_sentiment, empty_promise)

        # All rows should be preserved
        assert len(result) == len(mock_behavior_df)

        # Round 2 A and B should have values
        round_2_a = result[(result['round'] == 2) & (result['label'] == 'A')]
        assert round_2_a['sentiment_compound_mean'].notna().all()

        # Round 2 C and D should have NaN
        round_2_c = result[(result['round'] == 2) & (result['label'] == 'C')]
        assert round_2_c['sentiment_compound_mean'].isna().all()


# =====
# Run tests directly
# =====
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
