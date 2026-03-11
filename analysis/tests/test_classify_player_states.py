"""
Tests for player-level state classification (classify_player_states.py).
Author: Claude Code | Date: 2026-03-05
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'derived'))
sys.path.insert(0, str(Path(__file__).parent))

from classify_player_states import (
    PlayerCooperativeState, PlayerStateClassification,
    build_player_state_classification, _others_total_contribution,
    _is_others_cooperative,
)
from conftest import make_experiment_1sg, make_group, make_player, _make_round


# =====
# Others threshold boundary
# =====
class TestOthersThresholdBoundary:
    """Others' total contribution cooperative/noncooperative boundary."""

    def test_at_threshold_is_cooperative(self):
        """Others' total exactly 60 -> cooperative state."""
        exp = make_experiment_1sg([[("A", 25.0, 1), ("B", 20.0, 2), ("C", 20.0, 3), ("D", 20.0, 4)]])
        result = build_player_state_classification(exp, {})
        # For player A: others = 20+20+20 = 60 >= 60 -> cooperative
        assert result.cooperative.observation_count >= 1

    def test_above_threshold_is_cooperative(self):
        """Others' total above 60 -> cooperative state."""
        exp = make_experiment_1sg([[("A", 25.0, 1), ("B", 25.0, 2), ("C", 25.0, 3), ("D", 25.0, 4)]])
        result = build_player_state_classification(exp, {})
        # For each player: others = 75 >= 60 -> all cooperative
        assert result.cooperative.observation_count == 4
        assert result.noncooperative.observation_count == 0

    def test_below_threshold_is_noncooperative(self):
        """Others' total below 60 -> noncooperative state."""
        exp = make_experiment_1sg([[("A", 25.0, 1), ("B", 5.0, 2), ("C", 5.0, 3), ("D", 5.0, 4)]])
        result = build_player_state_classification(exp, {})
        # For player A: others = 5+5+5 = 15 < 60 -> noncooperative
        assert result.noncooperative.observation_count >= 1

    def test_custom_threshold(self):
        """Custom others_threshold shifts boundary."""
        exp = make_experiment_1sg([[("A", 25.0, 1), ("B", 15.0, 2), ("C", 15.0, 3), ("D", 15.0, 4)]])
        # For player A: others = 15+15+15 = 45
        high = build_player_state_classification(exp, {}, others_threshold=50.0)
        low = build_player_state_classification(exp, {}, others_threshold=45.0)
        assert high.noncooperative.observation_count >= 1  # 45 < 50
        assert low.cooperative.observation_count >= 1  # 45 >= 45


# =====
# Same group, different states (KEY differentiator from group-level)
# =====
class TestSameGroupDifferentStates:
    """Two players in the same group can have different states."""

    def test_two_players_different_states(self):
        """Player D cooperative, Player A noncooperative in same group."""
        exp = make_experiment_1sg([
            [("A", 25.0, 1), ("B", 20.0, 2), ("C", 20.0, 3), ("D", 0.0, 4)],
        ])
        result = build_player_state_classification(exp, {})
        # Player D: others = 25+20+20 = 65 >= 60 -> cooperative state
        # Player A: others = 20+20+0 = 40 < 60 -> noncooperative state
        assert result.cooperative.observation_count >= 1
        assert result.noncooperative.observation_count >= 1

    def test_all_same_state_when_symmetric(self):
        """All players cooperative when all contribute equally high."""
        exp = make_experiment_1sg([
            [("A", 25.0, 1), ("B", 25.0, 2), ("C", 25.0, 3), ("D", 25.0, 4)],
        ])
        result = build_player_state_classification(exp, {})
        # All others = 75 >= 60 -> all cooperative
        assert result.cooperative.observation_count == 4
        assert result.noncooperative.observation_count == 0


# =====
# Matrix cell assignment
# =====
class TestMatrixCellAssignment:
    """Observations land in correct matrix cells."""

    def test_cooperative_behavior_cooperative_state(self):
        """High contributor in cooperative state -> cooperative/no_promise cell."""
        exp = make_experiment_1sg([
            [("A", 25.0, 1), ("B", 25.0, 2), ("C", 25.0, 3), ("D", 25.0, 4)],
        ])
        result = build_player_state_classification(exp, {})
        assert result.cooperative.matrix.cooperative_no_promise.observation_count == 4

    def test_noncooperative_behavior_in_cooperative_state(self):
        """Low contributor whose others are cooperative -> noncoop behavior in coop state."""
        exp = make_experiment_1sg([
            [("A", 25.0, 1), ("B", 25.0, 2), ("C", 25.0, 3), ("D", 5.0, 4)],
        ])
        result = build_player_state_classification(exp, {})
        # Player D: others=75 >= 60 -> coop state, contribution=5 < 20 -> noncoop behavior
        assert result.cooperative.matrix.noncooperative_no_promise.observation_count >= 1

    def test_all_four_cells_populated(self):
        """All 4 matrix cells get observations with mixed behavior and promises."""
        exp = make_experiment_1sg([
            [("A", 25.0, 1), ("B", 25.0, 2), ("C", 5.0, 3), ("D", 5.0, 4)],
            [("A", 25.0, 1), ("B", 25.0, 2), ("C", 5.0, 3), ("D", 5.0, 4)],
        ])
        lookup = {
            ("s1", "supergame1", 2, "A"): True,
            ("s1", "supergame1", 2, "C"): True,
        }
        result = build_player_state_classification(exp, lookup)
        # Check at least one state has all four cells populated
        has_all_four = False
        for state in [result.cooperative, result.noncooperative]:
            m = state.matrix
            if all(cell.observation_count >= 1 for cell in m.cells):
                has_all_four = True
        assert has_all_four


# =====
# Player history
# =====
class TestPlayerHistory:
    """PlayerHistory tracks observations across rounds."""

    def test_player_history_created(self):
        """PlayerHistory exists after classification."""
        exp = make_experiment_1sg([[("A", 25.0, 1), ("B", 25.0, 2), ("C", 25.0, 3), ("D", 25.0, 4)]])
        result = build_player_state_classification(exp, {})
        found = _find_player_history(result, "s1:A")
        assert found is not None
        assert found.key == "s1:A"

    def test_observations_across_rounds(self):
        """Player accumulates observations across multiple rounds."""
        exp = make_experiment_1sg([
            [("A", 25.0, 1), ("B", 25.0, 2), ("C", 25.0, 3), ("D", 25.0, 4)],
            [("A", 20.0, 1), ("B", 20.0, 2), ("C", 20.0, 3), ("D", 20.0, 4)],
            [("A", 25.0, 1), ("B", 25.0, 2), ("C", 25.0, 3), ("D", 25.0, 4)],
        ])
        result = build_player_state_classification(exp, {})
        all_obs = _find_all_observations(result, "s1:A")
        assert len(all_obs) == 3


# =====
# Promise integration
# =====
class TestPromiseIntegration:
    """Promise lookup integrates correctly with player-level classification."""

    def test_round1_always_no_promise(self):
        """Round 1 players get made_promise=False even if lookup says True."""
        exp = make_experiment_1sg([
            [("A", 25.0, 1), ("B", 25.0, 2), ("C", 25.0, 3), ("D", 25.0, 4)],
        ])
        lookup = {("s1", "supergame1", 1, "A"): True}
        result = build_player_state_classification(exp, lookup)
        total_promise = sum(
            s.matrix.cooperative_promise.observation_count
            + s.matrix.noncooperative_promise.observation_count
            for s in [result.cooperative, result.noncooperative]
        )
        assert total_promise == 0

    def test_promise_from_lookup(self):
        """Round 2+ players use promise lookup value."""
        exp = make_experiment_1sg([
            [("A", 25.0, 1), ("B", 25.0, 2), ("C", 25.0, 3), ("D", 25.0, 4)],
            [("A", 25.0, 1), ("B", 25.0, 2), ("C", 25.0, 3), ("D", 25.0, 4)],
        ])
        lookup = {("s1", "supergame1", 2, "A"): True}
        result = build_player_state_classification(exp, lookup)
        total_promise = sum(
            s.matrix.cooperative_promise.observation_count
            for s in [result.cooperative, result.noncooperative]
        )
        assert total_promise == 1


# =====
# Summary statistics
# =====
class TestSummaryStats:
    """Summary statistics are correct."""

    def test_total_observations(self):
        """Total observations = players * rounds."""
        exp = make_experiment_1sg([
            [("A", 25.0, 1), ("B", 25.0, 2), ("C", 25.0, 3), ("D", 25.0, 4)],
            [("A", 25.0, 1), ("B", 25.0, 2), ("C", 25.0, 3), ("D", 25.0, 4)],
        ])
        result = build_player_state_classification(exp, {})
        total = result.cooperative.observation_count + result.noncooperative.observation_count
        assert total == 8  # 4 players * 2 rounds

    def test_behavior_counts_sum(self):
        """Behavior counts sum to total observations within each state."""
        exp = make_experiment_1sg([
            [("A", 25.0, 1), ("B", 25.0, 2), ("C", 5.0, 3), ("D", 5.0, 4)],
        ])
        result = build_player_state_classification(exp, {})
        for state in [result.cooperative, result.noncooperative]:
            bc = state.behavior_counts
            assert bc["cooperative"] + bc["noncooperative"] == state.observation_count

    def test_zero_contributor_count(self):
        """Zero contributor count matches contribution=0 observations."""
        exp = make_experiment_1sg([
            [("A", 25.0, 1), ("B", 25.0, 2), ("C", 0.0, 3), ("D", 25.0, 4)],
        ])
        result = build_player_state_classification(exp, {})
        total_zero = result.cooperative.zero_contributor_count + result.noncooperative.zero_contributor_count
        assert total_zero == 1

    def test_summary_string_contains_key_info(self):
        """Summary string includes thresholds and state labels."""
        exp = make_experiment_1sg([
            [("A", 25.0, 1), ("B", 25.0, 2), ("C", 25.0, 3), ("D", 25.0, 4)],
        ])
        summary = build_player_state_classification(exp, {}).summary()
        for expected in ["60.0", "20.0", "COOPERATIVE", "NONCOOPERATIVE",
                         "Cooperative behavior:", "Noncooperative behavior:",
                         "Zero contributors:"]:
            assert expected in summary

    def test_thresholds_stored(self):
        """Custom thresholds are stored in PlayerStateClassification."""
        exp = make_experiment_1sg([[("A", 25.0, 1), ("B", 25.0, 2), ("C", 25.0, 3), ("D", 25.0, 4)]])
        result = build_player_state_classification(exp, {}, others_threshold=50.0, player_threshold=15.0)
        assert result.others_threshold == 50.0
        assert result.player_threshold == 15.0


# =====
# Helper: _others_total_contribution
# =====
class TestOthersTotalContribution:
    """Direct tests for _others_total_contribution helper."""

    def test_excludes_target_player(self):
        """Sum excludes the specified player."""
        g = make_group(1, [("A", 25.0, 1), ("B", 10.0, 2), ("C", 5.0, 3)])
        assert _others_total_contribution(g, "A") == 15.0

    def test_single_player_group(self):
        """Single-player group -> others_total = 0."""
        g = make_group(1, [("A", 25.0, 1)])
        assert _others_total_contribution(g, "A") == 0.0


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


def _find_all_observations(classification, key):
    """Find all observations for a player key across all states/cells."""
    return [
        obs
        for state in [classification.cooperative, classification.noncooperative]
        for cell in state.matrix.cells
        if key in cell.players
        for obs in cell.players[key].observations
    ]
