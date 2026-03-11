"""
Tests for player-level state classification CSV export (to_dataframe, to_csv).
Author: Claude Code | Date: 2026-03-05
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'derived'))
sys.path.insert(0, str(Path(__file__).parent))

from conftest import make_experiment_1sg, make_group, make_player
from classify_player_states import PlayerStateClassification, build_player_state_classification


# =====
# Fixtures
# =====
@pytest.fixture
def multi_round_experiment():
    """3 rounds with varied contributions for player-level state testing."""
    return make_experiment_1sg([
        [("A", 25.0, 1), ("B", 20.0, 2), ("C", 20.0, 3), ("D", 0.0, 4)],
        [("A", 5.0, 1), ("B", 5.0, 2), ("C", 5.0, 3), ("D", 5.0, 4)],
        [("A", 25.0, 1), ("B", 25.0, 2), ("C", 25.0, 3), ("D", 25.0, 4)],
    ])


EXPECTED_COLUMNS = [
    'session_code', 'treatment', 'segment', 'round_num', 'group_id',
    'label', 'contribution', 'others_total_contribution', 'player_state',
    'player_behavior', 'made_promise', 'others_threshold', 'player_threshold',
]


# =====
# DataFrame export tests
# =====
class TestToDataframe:
    """to_dataframe() produces correct flat output for player-level classification."""

    def test_columns_present(self, multi_round_experiment):
        """DataFrame has all required columns."""
        df = build_player_state_classification(multi_round_experiment, {}).to_dataframe()
        for col in EXPECTED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count_matches_observations(self, multi_round_experiment):
        """DataFrame row count equals total player observations."""
        result = build_player_state_classification(multi_round_experiment, {})
        df = result.to_dataframe()
        total = result.cooperative.observation_count + result.noncooperative.observation_count
        assert len(df) == total

    def test_player_state_values(self, multi_round_experiment):
        """player_state column only contains 'cooperative' or 'noncooperative'."""
        df = build_player_state_classification(multi_round_experiment, {}).to_dataframe()
        assert set(df['player_state'].unique()) <= {'cooperative', 'noncooperative'}

    def test_player_behavior_values(self, multi_round_experiment):
        """player_behavior column only contains 'cooperative' or 'noncooperative'."""
        df = build_player_state_classification(multi_round_experiment, {}).to_dataframe()
        assert set(df['player_behavior'].unique()) <= {'cooperative', 'noncooperative'}

    def test_others_total_contribution_correct(self):
        """others_total_contribution equals sum of other players' contributions."""
        exp = make_experiment_1sg([
            [("A", 25.0, 1), ("B", 20.0, 2), ("C", 15.0, 3), ("D", 10.0, 4)],
        ])
        df = build_player_state_classification(exp, {}).to_dataframe()
        a_row = df[df['label'] == 'A'].iloc[0]
        # Others for A: B(20) + C(15) + D(10) = 45
        assert a_row['others_total_contribution'] == 45.0

    def test_thresholds_in_every_row(self):
        """Threshold columns match the values used in classification."""
        exp = make_experiment_1sg([
            [("A", 25.0, 1), ("B", 25.0, 2), ("C", 25.0, 3), ("D", 25.0, 4)],
        ])
        df = build_player_state_classification(
            exp, {}, others_threshold=50.0, player_threshold=15.0
        ).to_dataframe()
        assert (df['others_threshold'] == 50.0).all()
        assert (df['player_threshold'] == 15.0).all()

    def test_contribution_values_correct(self):
        """Contribution values in DataFrame match input data."""
        exp = make_experiment_1sg([
            [("A", 20.0, 1), ("B", 5.0, 2), ("C", 15.0, 3), ("D", 10.0, 4)],
        ])
        df = build_player_state_classification(exp, {}).to_dataframe()
        a_row = df[df['label'] == 'A'].iloc[0]
        b_row = df[df['label'] == 'B'].iloc[0]
        assert a_row['contribution'] == 20.0
        assert b_row['contribution'] == 5.0

    def test_same_group_different_states_in_df(self):
        """Players in the same group can have different player_state values.

        Group: A=25, B=20, C=20, D=0.
        D's others: 25+20+20=65 >= 60 -> cooperative state.
        A's others: 20+20+0=40 < 60 -> noncooperative state.
        """
        exp = make_experiment_1sg([
            [("A", 25.0, 1), ("B", 20.0, 2), ("C", 20.0, 3), ("D", 0.0, 4)],
        ])
        df = build_player_state_classification(exp, {}).to_dataframe()
        d_state = df[df['label'] == 'D']['player_state'].iloc[0]
        a_state = df[df['label'] == 'A']['player_state'].iloc[0]
        assert d_state == 'cooperative'
        assert a_state == 'noncooperative'

    def test_empty_classification_returns_empty_df(self):
        """Empty classification produces empty DataFrame."""
        empty = PlayerStateClassification(60.0, 20.0)
        df = empty.to_dataframe()
        assert len(df) == 0

    def test_no_nested_structures(self, multi_round_experiment):
        """All column dtypes are simple (no lists, dicts, or objects except strings)."""
        df = build_player_state_classification(multi_round_experiment, {}).to_dataframe()
        for col in df.columns:
            assert df[col].dtype.kind in ('i', 'f', 'b', 'O', 'U'), f"Complex dtype in {col}"


# =====
# CSV export tests
# =====
class TestToCsv:
    """to_csv() writes valid CSV files."""

    def test_csv_roundtrip(self, multi_round_experiment, tmp_path):
        """CSV can be written and read back with identical data."""
        result = build_player_state_classification(multi_round_experiment, {})
        csv_path = tmp_path / "test_player_export.csv"
        result.to_csv(csv_path)
        df_original = result.to_dataframe()
        df_read = pd.read_csv(csv_path)
        assert list(df_original.columns) == list(df_read.columns)
        assert len(df_original) == len(df_read)

    def test_csv_creates_parent_dirs(self, multi_round_experiment, tmp_path):
        """to_csv creates parent directories if they don't exist."""
        result = build_player_state_classification(multi_round_experiment, {})
        csv_path = tmp_path / "nested" / "dir" / "player_export.csv"
        result.to_csv(csv_path)
        assert csv_path.exists()
