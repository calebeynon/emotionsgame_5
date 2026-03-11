"""
Player-level state classification: cooperative/noncooperative based on others' contributions.
Author: Claude Code | Date: 2026-03-05
"""

import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiment_data import Experiment, Session
from classify_states import (
    DEFAULT_PLAYER_THRESHOLD, MatrixCell, Observation, PlayerHistory,
    TwoByTwoMatrix,
)
from classify_states_io import (
    PLAYER_OUTPUT_FILE, PROMISE_FILE, load_experiment, load_promise_lookup,
    get_promise, validate_contribution, player_obs_to_row,
)

# DEFAULT THRESHOLDS
DEFAULT_OTHERS_THRESHOLD = 60.0


# =====
# Main function
# =====
def main():
    """Load experiment data, build player-level classification, export CSV."""
    experiment = load_experiment()
    promise_lookup = load_promise_lookup(PROMISE_FILE)
    classification = build_player_state_classification(experiment, promise_lookup)
    classification.to_csv(PLAYER_OUTPUT_FILE)
    print(f"Wrote {PLAYER_OUTPUT_FILE}")
    print(classification.summary())


# =====
# Player-level cooperative state
# =====
class PlayerCooperativeState:
    """One side of the player-level cooperative/noncooperative classification."""

    def __init__(self, label: str):
        self.label = label
        self.matrix = TwoByTwoMatrix()
        self.others_totals: Dict[str, float] = {}

    @property
    def observation_count(self) -> int:
        """Total player observations across all matrix cells."""
        return sum(cell.observation_count for cell in self.matrix.cells)

    @property
    def behavior_counts(self) -> Dict[str, int]:
        """Player observation counts by behavior (summed across promise axis)."""
        coop = (self.matrix.cooperative_promise.observation_count
                + self.matrix.cooperative_no_promise.observation_count)
        noncoop = (self.matrix.noncooperative_promise.observation_count
                   + self.matrix.noncooperative_no_promise.observation_count)
        return {"cooperative": coop, "noncooperative": noncoop}

    @property
    def zero_contributor_count(self) -> int:
        """Number of player-round observations with contribution == 0."""
        return sum(
            1 for cell in self.matrix.cells
            for ph in cell.players.values()
            for obs in ph.observations if obs.contribution == 0
        )


# =====
# Top-level classification container
# =====
class PlayerStateClassification:
    """Top-level container for player-level state classification."""

    def __init__(self, others_threshold: float, player_threshold: float):
        self.others_threshold = others_threshold
        self.player_threshold = player_threshold
        self.cooperative = PlayerCooperativeState("cooperative")
        self.noncooperative = PlayerCooperativeState("noncooperative")

    def summary(self) -> str:
        """Human-readable summary of player-level classification."""
        return _build_summary(self)

    def to_dataframe(self) -> pd.DataFrame:
        """Export classification to a flat DataFrame, one row per player-round."""
        rows = []
        for state in [self.cooperative, self.noncooperative]:
            for cell in state.matrix.cells:
                for ph in cell.players.values():
                    for obs in ph.observations:
                        key = _obs_key(obs)
                        others = state.others_totals.get(key, 0.0)
                        rows.append(player_obs_to_row(obs, state, cell, others, self))
        return pd.DataFrame(rows)

    def to_csv(self, filepath: Path):
        """Write classification DataFrame to CSV."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(filepath, index=False)


# =====
# Builder
# =====
def build_player_state_classification(
    experiment: Experiment,
    promise_lookup: dict,
    others_threshold: float = DEFAULT_OTHERS_THRESHOLD,
    player_threshold: float = DEFAULT_PLAYER_THRESHOLD,
) -> PlayerStateClassification:
    """Build player-level state classification from experiment data."""
    classification = PlayerStateClassification(others_threshold, player_threshold)
    for session_code, session in experiment.sessions.items():
        _classify_session(
            session_code, session, promise_lookup,
            others_threshold, player_threshold, classification
        )
    return classification


# =====
# Session and round classification helpers
# =====
def _classify_session(session_code, session, promise_lookup,
                      others_threshold, player_threshold, classification):
    """Classify all supergame player-rounds in a session."""
    for segment_name, segment in session.segments.items():
        if not segment_name.startswith('supergame'):
            continue
        for round_num in sorted(segment.rounds.keys()):
            round_obj = segment.get_round(round_num)
            for group_id, group in round_obj.groups.items():
                _classify_group_players(
                    session_code, session, segment_name, round_num,
                    group_id, group, promise_lookup,
                    others_threshold, player_threshold, classification
                )


def _classify_group_players(session_code, session, segment_name, round_num,
                            group_id, group, promise_lookup,
                            others_threshold, player_threshold, cls):
    """Classify each player in a group-round based on others' contributions."""
    for label, player in group.players.items():
        _classify_player_round(
            session_code, session, segment_name, round_num,
            group_id, label, player, group, promise_lookup,
            others_threshold, player_threshold, cls
        )


def _others_total_contribution(group, exclude_label) -> float:
    """Sum contributions of all group members except the excluded player."""
    return sum(
        p.contribution
        for lbl, p in group.players.items() if lbl != exclude_label
    )


def _is_others_cooperative(group, exclude_label, threshold) -> tuple:
    """Check if others' total contribution meets threshold. Returns (total, is_coop)."""
    total = _others_total_contribution(group, exclude_label)
    return total, total >= threshold


def _classify_player_round(session_code, session, segment_name, round_num,
                           group_id, label, player, group, promise_lookup,
                           others_threshold, player_threshold, cls):
    """Classify a single player-round into the appropriate state and matrix cell."""
    others_total, is_coop = _is_others_cooperative(group, label, others_threshold)
    state = cls.cooperative if is_coop else cls.noncooperative
    contribution = validate_contribution(player, label, session_code, segment_name, round_num)
    made_promise = get_promise(promise_lookup, session_code, segment_name, round_num, label)
    behavior = "cooperative" if contribution >= player_threshold else "noncooperative"
    promise_axis = "promise" if made_promise else "no_promise"
    obs = Observation(session_code, session.treatment, segment_name, round_num,
                      group_id, label, contribution, made_promise, player)
    state.matrix[(behavior, promise_axis)].add_observation(obs, session)
    state.others_totals[_obs_key(obs)] = others_total


# =====
# Output helpers
# =====
def _obs_key(obs: Observation) -> str:
    """Build unique key for an observation to index others_totals."""
    return f"{obs.session_code}:{obs.segment}:{obs.round_num}:{obs.group_id}:{obs.label}"


def _build_summary(cls: PlayerStateClassification) -> str:
    """Build human-readable summary string."""
    sep = "=" * 50
    total_obs = cls.cooperative.observation_count + cls.noncooperative.observation_count
    lines = [sep, "PLAYER STATE CLASSIFICATION SUMMARY", sep,
             f"Others threshold: {cls.others_threshold}",
             f"Player threshold: {cls.player_threshold}",
             f"Total player observations: {total_obs}"]
    for state in [cls.cooperative, cls.noncooperative]:
        bc = state.behavior_counts
        lines.append(f"\n{state.label.upper()} STATE:")
        lines.append(f"  Player observations: {state.observation_count}")
        lines.append(f"  Cooperative behavior: {bc['cooperative']}")
        lines.append(f"  Noncooperative behavior: {bc['noncooperative']}")
        lines.append(f"  Zero contributors: {state.zero_contributor_count}")
        for cell in state.matrix.cells:
            lines.append(f"  {cell.behavior_label}/{cell.promise_label}: "
                         f"{cell.count} players, {cell.observation_count} obs")
    lines.append(sep)
    return "\n".join(lines)


# %%
if __name__ == "__main__":
    main()
