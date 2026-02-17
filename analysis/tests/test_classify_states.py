"""
Tests for state classification module (classify_states.py).
Author: Claude Code | Date: 2026-02-16
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'derived'))

from classify_states import (
    MatrixCell, Observation, TwoByTwoMatrix,
    _build_lookup_from_df, _get_promise, build_state_classification,
)
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
    """Build single-session, single-supergame experiment from rounds data.

    Args:
        rounds_data: list of lists of (label, contribution, pid) per round.
    """
    seg = Segment("supergame1")
    for i, players in enumerate(rounds_data, 1):
        seg.add_round(_make_round(i, [make_group(1, players)]))
    sess = Session("s1", 1)
    sess.add_segment(seg)
    for rnd in seg.rounds.values():
        for label, player in rnd.players.items():
            sess.participant_labels[player.participant_id] = label
    exp = Experiment(name="Test")
    exp.add_session(sess)
    return exp


def _make_round(num, groups):
    """Create a Round from groups."""
    r = Round(num)
    for g in groups:
        r.add_group(g)
    return r


# =====
# Shared fixture
# =====
@pytest.fixture
def multi_round_experiment():
    """3 rounds: cooperative(80%), noncooperative(20%), cooperative(100%)."""
    return make_experiment_1sg([
        [("A", 25.0, 1), ("B", 20.0, 2), ("C", 15.0, 3), ("D", 20.0, 4)],
        [("A", 5.0, 1), ("B", 5.0, 2), ("C", 5.0, 3), ("D", 5.0, 4)],
        [("A", 25.0, 1), ("B", 25.0, 2), ("C", 25.0, 3), ("D", 25.0, 4)],
    ])


# =====
# Group classification boundary
# =====
class TestGroupBoundary:
    """Group cooperative/noncooperative boundary at threshold."""

    def test_at_threshold_is_cooperative(self):
        """Group mean at exactly 50% of endowment is cooperative."""
        exp = make_experiment_1sg([[("A", 25.0, 1), ("B", 0.0, 2)]])
        result = build_state_classification(exp, {})
        assert result.cooperative.group_count == 1
        assert result.noncooperative.group_count == 0

    def test_above_threshold_is_cooperative(self):
        """Group mean above 50% is cooperative."""
        exp = make_experiment_1sg([[("A", 25.0, 1), ("B", 25.0, 2)]])
        result = build_state_classification(exp, {})
        assert result.cooperative.group_count == 1

    def test_below_threshold_is_noncooperative(self):
        """Group mean below 50% is noncooperative (mean=12, pct=48%)."""
        exp = make_experiment_1sg([[("A", 24.0, 1), ("B", 0.0, 2)]])
        result = build_state_classification(exp, {})
        assert result.noncooperative.group_count == 1
        assert result.cooperative.group_count == 0


# =====
# Player classification boundary
# =====
class TestPlayerBoundary:
    """Player cooperative/noncooperative boundary at threshold."""

    def test_at_threshold_is_cooperative(self):
        """Player contribution exactly at 12.5 is cooperative."""
        exp = make_experiment_1sg([[("A", 12.5, 1), ("B", 25.0, 2)]])
        result = build_state_classification(exp, {})
        assert result.cooperative.matrix.cooperative_no_promise.observation_count >= 1

    def test_below_threshold_is_noncooperative(self):
        """Player contribution below 12.5 is noncooperative."""
        exp = make_experiment_1sg([[("A", 12.0, 1), ("B", 25.0, 2)]])
        result = build_state_classification(exp, {})
        assert result.cooperative.matrix.noncooperative_no_promise.observation_count >= 1


# =====
# Matrix cell assignment
# =====
class TestMatrixCells:
    """All 4 matrix cells receive correct observations."""

    def test_all_four_cells_populated(self):
        """All 4 cells get observations with mixed behavior and promise data."""
        exp = make_experiment_1sg([
            [("A", 25.0, 1), ("B", 25.0, 2), ("C", 0.0, 3), ("D", 0.0, 4)],
            [("A", 25.0, 1), ("B", 25.0, 2), ("C", 0.0, 3), ("D", 0.0, 4)],
        ])
        lookup = {
            ("s1", "supergame1", 2, "A"): True,
            ("s1", "supergame1", 2, "C"): True,
        }
        result = build_state_classification(exp, lookup)
        m = result.cooperative.matrix
        assert m.cooperative_promise.observation_count >= 1
        assert m.cooperative_no_promise.observation_count >= 1
        assert m.noncooperative_promise.observation_count >= 1
        assert m.noncooperative_no_promise.observation_count >= 1


# =====
# PlayerHistory
# =====
class TestPlayerHistory:
    """PlayerHistory contains full trajectory."""

    def test_full_trajectory(self):
        """PlayerHistory.rounds_data spans all supergame rounds."""
        exp = make_experiment_1sg([
            [("A", 25.0, 1)], [("A", 10.0, 1)], [("A", 5.0, 1)],
        ])
        result = build_state_classification(exp, {})
        ph = _find_player_history(result, "s1:A")
        assert ph is not None
        assert ph.total_rounds == 3
        assert ph.key == "s1:A"

    def test_mean_contribution_across_cells(self):
        """Mean contribution computed across all cells for a player."""
        exp = make_experiment_1sg([[("A", 20.0, 1)], [("A", 10.0, 1)]])
        all_ph = _find_all_player_histories(build_state_classification(exp, {}), "s1:A")
        all_obs = [o for ph in all_ph for o in ph.observations]
        assert sum(o.contribution for o in all_obs) / len(all_obs) == 15.0


# =====
# Same player in multiple cells
# =====
class TestPlayerMultipleCells:
    """Same player can appear in different cells across rounds."""

    def test_same_player_multiple_cells(self):
        """Player appears in different matrix cells across rounds."""
        exp = make_experiment_1sg([
            [("A", 25.0, 1), ("B", 25.0, 2)],
            [("A", 0.0, 1), ("B", 25.0, 2)],
        ])
        lookup = {("s1", "supergame1", 2, "A"): True}
        result = build_state_classification(exp, lookup)
        assert len(_find_cells_with_player(result, "s1:A")) >= 2


# =====
# Promise lookup integration
# =====
class TestPromiseLookup:
    """Promise lookup integration with classification."""

    def test_build_from_dataframe(self):
        """_build_lookup_from_df correctly builds lookup dict."""
        df = pd.DataFrame({
            'session_code': ['s1', 's1'], 'segment': ['supergame1'] * 2,
            'round': [2, 2], 'label': ['A', 'B'], 'promise_count': [1, 0],
        })
        lookup = _build_lookup_from_df(df)
        assert lookup[('s1', 'supergame1', 2, 'A')] is True
        assert lookup[('s1', 'supergame1', 2, 'B')] is False

    def test_missing_defaults_false(self):
        """Players not in lookup default to no promise."""
        exp = make_experiment_1sg([[("A", 25.0, 1)], [("A", 25.0, 1)]])
        result = build_state_classification(exp, {})
        for state in [result.cooperative, result.noncooperative]:
            assert state.matrix.cooperative_promise.observation_count == 0
            assert state.matrix.noncooperative_promise.observation_count == 0


# =====
# Round 1 promise behavior
# =====
class TestRound1Promise:
    """Round 1 always has made_promise=False."""

    def test_round1_always_no_promise(self):
        """Round 1 players get made_promise=False even if lookup says True."""
        exp = make_experiment_1sg([[("A", 25.0, 1)]])
        lookup = {("s1", "supergame1", 1, "A"): True}
        result = build_state_classification(exp, lookup)
        total = sum(
            s.matrix.cooperative_promise.observation_count
            + s.matrix.noncooperative_promise.observation_count
            for s in [result.cooperative, result.noncooperative]
        )
        assert total == 0

    def test_get_promise_round1_returns_false(self):
        """_get_promise returns False for round 1."""
        assert _get_promise({("s1", "sg1", 1, "A"): True}, "s1", "sg1", 1, "A") is False

    def test_get_promise_round2_uses_lookup(self):
        """_get_promise returns lookup value for round > 1."""
        assert _get_promise({("s1", "sg1", 2, "A"): True}, "s1", "sg1", 2, "A") is True


# =====
# Subscript access
# =====
class TestSubscriptAccess:
    """Matrix subscript access matches named attributes."""

    def test_subscript_matches_named(self):
        """TwoByTwoMatrix[key] returns same object as named attribute."""
        m = TwoByTwoMatrix()
        assert m[("cooperative", "promise")] is m.cooperative_promise
        assert m[("cooperative", "no_promise")] is m.cooperative_no_promise
        assert m[("noncooperative", "promise")] is m.noncooperative_promise
        assert m[("noncooperative", "no_promise")] is m.noncooperative_no_promise

    def test_cells_returns_four(self):
        """TwoByTwoMatrix.cells returns all 4 cells."""
        assert len(TwoByTwoMatrix().cells) == 4


# =====
# Summary statistics
# =====
class TestSummaryStats:
    """Summary statistics are correct."""

    def test_total_observations(self, multi_round_experiment):
        """Total observations = players * rounds."""
        result = build_state_classification(multi_round_experiment, {})
        total = result.cooperative.observation_count + result.noncooperative.observation_count
        assert total == 12  # 4 players * 3 rounds

    def test_group_counts(self, multi_round_experiment):
        """Total group-rounds match expected count."""
        result = build_state_classification(multi_round_experiment, {})
        total = result.cooperative.group_count + result.noncooperative.group_count
        assert total == 3  # 1 group * 3 rounds

    def test_summary_string(self, multi_round_experiment):
        """Summary string includes thresholds and state labels."""
        summary = build_state_classification(multi_round_experiment, {}).summary()
        for expected in ["50.0%", "12.5", "COOPERATIVE", "NONCOOPERATIVE"]:
            assert expected in summary

    def test_cell_mean_contribution(self):
        """MatrixCell.mean_contribution calculates correctly."""
        cell = MatrixCell("cooperative", "promise")
        p = make_player("A", 20.0, 1)
        seg = Segment("sg1")
        seg.add_round(_make_round(1, [make_group(1, [("A", 20.0, 1)])]))
        sess = Session("s1", 1)
        sess.add_segment(seg)
        obs1 = Observation("s1", 1, "sg1", 1, 1, "A", 20.0, True, p)
        obs2 = Observation("s1", 1, "sg1", 2, 1, "A", 10.0, True, p)
        cell.add_observation(obs1, sess)
        cell.add_observation(obs2, sess)
        assert cell.mean_contribution == 15.0


# =====
# Custom thresholds
# =====
class TestCustomThresholds:
    """Custom thresholds work correctly."""

    def test_custom_group_threshold(self):
        """Custom group threshold changes classification boundary."""
        exp = make_experiment_1sg([[("A", 25.0, 1), ("B", 0.0, 2)]])
        assert build_state_classification(exp, {}, group_threshold=51.0).noncooperative.group_count == 1
        assert build_state_classification(exp, {}, group_threshold=50.0).cooperative.group_count == 1

    def test_custom_player_threshold(self):
        """Custom player threshold changes player classification."""
        exp = make_experiment_1sg([[("A", 10.0, 1), ("B", 25.0, 2)]])
        low = _find_cells_with_player(build_state_classification(exp, {}, player_threshold=10.0), "s1:A")
        high = _find_cells_with_player(build_state_classification(exp, {}, player_threshold=11.0), "s1:A")
        assert any(c.behavior_label == "cooperative" for c in low)
        assert any(c.behavior_label == "noncooperative" for c in high)

    def test_thresholds_stored(self):
        """Custom thresholds are stored in StateClassification."""
        exp = make_experiment_1sg([[("A", 25.0, 1)]])
        result = build_state_classification(exp, {}, group_threshold=75.0, player_threshold=20.0)
        assert result.group_threshold == 75.0
        assert result.player_threshold == 20.0


# =====
# Test helpers
# =====
def _find_player_history(classification, key):
    """Find first PlayerHistory by key across all states and cells."""
    for state in [classification.cooperative, classification.noncooperative]:
        for cell in state.matrix.cells:
            if key in cell.players:
                return cell.players[key]
    return None


def _find_all_player_histories(classification, key):
    """Find all PlayerHistory instances for a key across all states/cells."""
    return [
        cell.players[key]
        for state in [classification.cooperative, classification.noncooperative]
        for cell in state.matrix.cells
        if key in cell.players
    ]


def _find_cells_with_player(classification, key):
    """Find all MatrixCells containing a player by key."""
    return [
        cell
        for state in [classification.cooperative, classification.noncooperative]
        for cell in state.matrix.cells
        if key in cell.players
    ]
