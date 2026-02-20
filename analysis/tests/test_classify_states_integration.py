"""
Integration tests for classify_states against raw CSV data.

Verifies that StateClassification produces correct group/player
classifications by cross-referencing raw oTree exports and
promise_classifications.csv.

Author: Claude Code
Date: 2026-02-20
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'derived'))

from classify_states import (
    DEFAULT_GROUP_THRESHOLD,
    DEFAULT_PLAYER_THRESHOLD,
    build_state_classification,
)
from classify_states_io import (
    PROMISE_FILE,
    load_experiment,
    load_promise_lookup,
)

# CONSTANTS
TOTAL_SESSIONS = 10
PLAYERS_PER_SESSION = 16
ROUNDS_PER_SUPERGAME = {1: 3, 2: 4, 3: 3, 4: 7, 5: 5}
TOTAL_ROUNDS = sum(ROUNDS_PER_SUPERGAME.values())  # 22
TOTAL_OBSERVATIONS = TOTAL_SESSIONS * PLAYERS_PER_SESSION * TOTAL_ROUNDS  # 3520
GROUPS_PER_SESSION = 4
TOTAL_GROUP_ROUNDS = TOTAL_SESSIONS * GROUPS_PER_SESSION * TOTAL_ROUNDS  # 880


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def experiment():
    """Load the full experiment once for all tests."""
    return load_experiment()


@pytest.fixture(scope="module")
def promise_lookup():
    """Load promise lookup once for all tests."""
    return load_promise_lookup(PROMISE_FILE)


@pytest.fixture(scope="module")
def classification(experiment, promise_lookup):
    """Build state classification once for all tests."""
    return build_state_classification(experiment, promise_lookup)


# =====
# Test: total observation counts
# =====
def test_total_observation_count(classification):
    """Total player-round observations equals 10 sessions * 16 players * 22 rounds."""
    coop = classification.cooperative.observation_count
    noncoop = classification.noncooperative.observation_count
    assert coop + noncoop == TOTAL_OBSERVATIONS


def test_total_group_round_count(classification):
    """Total group-rounds equals 10 sessions * 4 groups * 22 rounds."""
    coop = classification.cooperative.group_count
    noncoop = classification.noncooperative.group_count
    assert coop + noncoop == TOTAL_GROUP_ROUNDS


# =====
# Test: conservation (4 cells partition each state's observations)
# =====
def test_matrix_cells_partition_observations(classification):
    """Each state's 4 cells should partition all observations in that state."""
    for state in [classification.cooperative, classification.noncooperative]:
        cell_total = sum(c.observation_count for c in state.matrix.cells)
        assert cell_total == state.observation_count


# =====
# Test: spot-check contributions against raw CSV values
# =====
def test_spot_check_contributions(classification):
    """Verify specific player contributions match raw CSV data."""
    cases = [
        ('sa7mprty', 'supergame1', 1, 'A', 15.0),
        ('sa7mprty', 'supergame1', 1, 'J', 5.0),
        ('sa7mprty', 'supergame1', 2, 'E', 25.0),
    ]
    for session, segment, rnd, label, expected in cases:
        obs = _find_observation(classification, session, segment, rnd, label)
        assert obs is not None, f"Observation not found: {session} {segment} R{rnd} {label}"
        assert obs.contribution == expected, f"{label} R{rnd}: {obs.contribution} != {expected}"


# =====
# Test: group cooperative/noncooperative classification
# =====
def test_noncooperative_group_sa7mprty_sg1_r1_g1(classification):
    """Group 1 SG1 R1 in sa7mprty: mean=9.0 (36% of 25) => noncooperative."""
    go = _find_group_obs(classification, 'sa7mprty', 'supergame1', 1, 1)
    assert go is not None, "Group observation not found"
    assert go.mean_contribution == 9.0
    assert go.is_cooperative is False


def test_cooperative_group_sa7mprty_sg1_r2_g1(classification):
    """Group 1 SG1 R2 in sa7mprty: all 25 (100% of 25) => cooperative."""
    go = _find_group_obs(classification, 'sa7mprty', 'supergame1', 2, 1)
    assert go is not None, "Group observation not found"
    assert go.mean_contribution == 25.0
    assert go.is_cooperative is True


def test_noncooperative_group_sa7mprty_sg1_r1_g4(classification):
    """Group 4 SG1 R1 in sa7mprty: mean=11.25 (45% of 25) => noncooperative."""
    go = _find_group_obs(classification, 'sa7mprty', 'supergame1', 1, 4)
    assert go is not None, "Group observation not found"
    assert go.mean_contribution == 11.25
    assert go.is_cooperative is False


# =====
# Test: player behavior classification
# =====
def test_player_cooperative_behavior(classification):
    """Player A (contrib=15 >= 12.5) in noncooperative group => noncooperative state, cooperative behavior."""
    obs = _find_observation(classification, 'sa7mprty', 'supergame1', 1, 'A')
    assert obs is not None
    # In noncooperative state (group is noncooperative)
    cell = _find_cell_for_obs(classification.noncooperative, obs)
    assert cell is not None
    assert cell.behavior_label == "cooperative"


def test_player_noncooperative_behavior(classification):
    """Player J (contrib=5 < 12.5) in noncooperative group => noncooperative state, noncooperative behavior."""
    obs = _find_observation(classification, 'sa7mprty', 'supergame1', 1, 'J')
    assert obs is not None
    cell = _find_cell_for_obs(classification.noncooperative, obs)
    assert cell is not None
    assert cell.behavior_label == "noncooperative"


# =====
# Test: promise classification
# =====
def test_round1_no_promise(classification):
    """Round 1 always has no promise (no prior chat to promise in)."""
    obs = _find_observation(classification, 'sa7mprty', 'supergame1', 1, 'A')
    assert obs is not None
    assert obs.made_promise is False


def test_promise_from_csv(classification):
    """Player E in sa7mprty SG1 R2 had promise_count=1 => made_promise=True."""
    obs = _find_observation(classification, 'sa7mprty', 'supergame1', 2, 'E')
    assert obs is not None
    assert obs.made_promise is True


def test_no_promise_from_csv(classification):
    """Player A in sa7mprty SG1 R2 had promise_count=0 => made_promise=False."""
    obs = _find_observation(classification, 'sa7mprty', 'supergame1', 2, 'A')
    assert obs is not None
    assert obs.made_promise is False


def test_promise_lookup_matches_csv(promise_lookup):
    """Spot-check promise lookup against known CSV values."""
    # Player J in sa7mprty SG1 R2 had promise_count=2 => True
    assert promise_lookup[('sa7mprty', 'supergame1', 2, 'J')] is True
    # Player K in sa7mprty SG1 R2 had promise_count=0 => False
    assert promise_lookup[('sa7mprty', 'supergame1', 2, 'K')] is False


# =====
# Test: PlayerHistory trajectory
# =====
def test_player_history_total_rounds(classification):
    """PlayerHistory.total_rounds includes all segments (intro + 5 SGs + final = 24)."""
    ph = _find_any_player_history(classification, 'sa7mprty', 'A')
    assert ph is not None
    # 22 supergame rounds + 1 introduction + 1 finalresults = 24
    assert ph.total_rounds == 24


def test_player_history_has_all_observations(classification):
    """Player A in sa7mprty appears in exactly 22 observations across all cells."""
    total_obs = _count_player_observations(classification, 'sa7mprty', 'A')
    assert total_obs == TOTAL_ROUNDS


# =====
# Test: thresholds and metadata
# =====
def test_default_thresholds(classification):
    """Classification uses the expected default thresholds."""
    assert classification.group_threshold == DEFAULT_GROUP_THRESHOLD
    assert classification.player_threshold == DEFAULT_PLAYER_THRESHOLD


def test_treatment_in_observations(classification):
    """Treatment values in observations match the file naming convention."""
    obs = _find_observation(classification, 'sa7mprty', 'supergame1', 1, 'A')
    assert obs is not None
    assert obs.treatment == 1  # 01_t1 file


def test_all_sessions_have_observations(classification, experiment):
    """Every loaded session contributes observations to the classification."""
    sessions_with_obs = set()
    for state in [classification.cooperative, classification.noncooperative]:
        for go in state.group_observations:
            sessions_with_obs.add(go.session_code)
    assert sessions_with_obs == set(experiment.sessions.keys())


# =====
# Helpers
# =====
def _find_observation(classification, session_code, segment, round_num, label):
    """Find a specific observation across all states and cells."""
    for state in [classification.cooperative, classification.noncooperative]:
        for cell in state.matrix.cells:
            for ph in cell.players.values():
                for obs in ph.observations:
                    if (obs.session_code == session_code
                            and obs.segment == segment
                            and obs.round_num == round_num
                            and obs.label == label):
                        return obs
    return None


def _find_group_obs(classification, session_code, segment, round_num, group_id):
    """Find a specific group observation across both states."""
    for state in [classification.cooperative, classification.noncooperative]:
        for go in state.group_observations:
            if (go.session_code == session_code
                    and go.segment == segment
                    and go.round_num == round_num
                    and go.group_id == group_id):
                return go
    return None


def _find_cell_for_obs(state, obs):
    """Find which matrix cell contains a specific observation."""
    for cell in state.matrix.cells:
        for ph in cell.players.values():
            if obs in ph.observations:
                return cell
    return None


def _find_any_player_history(classification, session_code, label):
    """Find a PlayerHistory for a given player across all cells."""
    key = f"{session_code}:{label}"
    for state in [classification.cooperative, classification.noncooperative]:
        for cell in state.matrix.cells:
            if key in cell.players:
                return cell.players[key]
    return None


def _count_player_observations(classification, session_code, label):
    """Count total observations for a player across all states and cells."""
    key = f"{session_code}:{label}"
    total = 0
    for state in [classification.cooperative, classification.noncooperative]:
        for cell in state.matrix.cells:
            if key in cell.players:
                total += len(cell.players[key].observations)
    return total
