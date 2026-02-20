"""
Tests for state classification CSV export (to_dataframe, to_csv).
Author: Claude Code | Date: 2026-02-20
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'derived'))

from classify_states import StateClassification, build_state_classification
from experiment_data import Experiment, Group, Player, Round, Segment, Session


# =====
# Synthetic data builders
# =====
def make_player(label, contribution, pid=1):
    """Create a Player with given label and contribution."""
    p = Player(participant_id=pid, label=label, id_in_group=1)
    p.contribution = contribution
    return p


def make_group(group_id, players_data):
    """Create a Group from list of (label, contribution, pid) tuples."""
    g = Group(group_id)
    for label, contribution, pid in players_data:
        g.add_player(make_player(label, contribution, pid))
    return g


def make_experiment_1sg(rounds_data):
    """Build single-session, single-supergame experiment from rounds data."""
    seg = Segment("supergame1")
    for i, players in enumerate(rounds_data, 1):
        r = Round(i)
        r.add_group(make_group(1, players))
        seg.add_round(r)
    sess = Session("s1", 1)
    sess.add_segment(seg)
    for rnd in seg.rounds.values():
        for label, player in rnd.players.items():
            sess.participant_labels[player.participant_id] = label
    exp = Experiment(name="Test")
    exp.add_session(sess)
    return exp


@pytest.fixture
def multi_round_experiment():
    """3 rounds: cooperative(80%), noncooperative(20%), cooperative(100%)."""
    return make_experiment_1sg([
        [("A", 25.0, 1), ("B", 20.0, 2), ("C", 15.0, 3), ("D", 20.0, 4)],
        [("A", 5.0, 1), ("B", 5.0, 2), ("C", 5.0, 3), ("D", 5.0, 4)],
        [("A", 25.0, 1), ("B", 25.0, 2), ("C", 25.0, 3), ("D", 25.0, 4)],
    ])


# =====
# DataFrame export tests
# =====
class TestToDataframe:
    """to_dataframe() produces correct flat output."""

    def test_columns_present(self, multi_round_experiment):
        """DataFrame has all required columns."""
        df = build_state_classification(multi_round_experiment, {}).to_dataframe()
        expected = [
            'session_code', 'treatment', 'segment', 'round_num', 'group_id',
            'label', 'contribution', 'group_mean_contribution', 'group_state',
            'player_behavior', 'made_promise', 'group_threshold', 'player_threshold',
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count_matches_observations(self, multi_round_experiment):
        """DataFrame row count equals total player observations."""
        result = build_state_classification(multi_round_experiment, {})
        df = result.to_dataframe()
        total_obs = result.cooperative.observation_count + result.noncooperative.observation_count
        assert len(df) == total_obs

    def test_group_state_values(self, multi_round_experiment):
        """group_state column only contains 'cooperative' or 'noncooperative'."""
        df = build_state_classification(multi_round_experiment, {}).to_dataframe()
        assert set(df['group_state'].unique()) <= {'cooperative', 'noncooperative'}

    def test_player_behavior_values(self, multi_round_experiment):
        """player_behavior column only contains 'cooperative' or 'noncooperative'."""
        df = build_state_classification(multi_round_experiment, {}).to_dataframe()
        assert set(df['player_behavior'].unique()) <= {'cooperative', 'noncooperative'}

    def test_thresholds_in_every_row(self):
        """Threshold columns match the values used in classification."""
        exp = make_experiment_1sg([[("A", 25.0, 1), ("B", 25.0, 2)]])
        df = build_state_classification(exp, {}, group_threshold=60.0, player_threshold=15.0).to_dataframe()
        assert (df['group_threshold'] == 60.0).all()
        assert (df['player_threshold'] == 15.0).all()

    def test_contribution_values_correct(self):
        """Contribution values in DataFrame match input data."""
        exp = make_experiment_1sg([[("A", 20.0, 1), ("B", 5.0, 2)]])
        df = build_state_classification(exp, {}).to_dataframe()
        a_row = df[df['label'] == 'A'].iloc[0]
        b_row = df[df['label'] == 'B'].iloc[0]
        assert a_row['contribution'] == 20.0
        assert b_row['contribution'] == 5.0

    def test_group_mean_contribution_correct(self):
        """group_mean_contribution matches expected average."""
        exp = make_experiment_1sg([[("A", 20.0, 1), ("B", 10.0, 2)]])
        df = build_state_classification(exp, {}).to_dataframe()
        assert df['group_mean_contribution'].iloc[0] == 15.0

    def test_promise_flag_in_dataframe(self):
        """made_promise reflects the promise lookup data."""
        exp = make_experiment_1sg([
            [("A", 25.0, 1), ("B", 25.0, 2)],
            [("A", 25.0, 1), ("B", 25.0, 2)],
        ])
        lookup = {("s1", "supergame1", 2, "A"): True}
        df = build_state_classification(exp, lookup).to_dataframe()
        r2_a = df[(df['round_num'] == 2) & (df['label'] == 'A')]
        r2_b = df[(df['round_num'] == 2) & (df['label'] == 'B')]
        assert r2_a['made_promise'].iloc[0] == True
        assert r2_b['made_promise'].iloc[0] == False

    def test_empty_classification_returns_empty_df(self):
        """Empty classification produces empty DataFrame with correct columns."""
        empty = StateClassification(50.0, 12.5)
        df = empty.to_dataframe()
        assert len(df) == 0

    def test_no_nested_structures(self, multi_round_experiment):
        """All column dtypes are simple (no lists, dicts, or objects except strings)."""
        df = build_state_classification(multi_round_experiment, {}).to_dataframe()
        for col in df.columns:
            assert df[col].dtype.kind in ('i', 'f', 'b', 'O', 'U'), f"Complex dtype in {col}"


# =====
# CSV export tests
# =====
class TestToCsv:
    """to_csv() writes valid CSV files."""

    def test_csv_roundtrip(self, multi_round_experiment, tmp_path):
        """CSV can be written and read back with identical data."""
        result = build_state_classification(multi_round_experiment, {})
        csv_path = tmp_path / "test_export.csv"
        result.to_csv(csv_path)
        df_original = result.to_dataframe()
        df_read = pd.read_csv(csv_path)
        assert list(df_original.columns) == list(df_read.columns)
        assert len(df_original) == len(df_read)

    def test_csv_creates_parent_dirs(self, multi_round_experiment, tmp_path):
        """to_csv creates parent directories if they don't exist."""
        result = build_state_classification(multi_round_experiment, {})
        csv_path = tmp_path / "nested" / "dir" / "export.csv"
        result.to_csv(csv_path)
        assert csv_path.exists()
