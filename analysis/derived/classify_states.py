"""
State classification: cooperative/noncooperative group states with promise axis.
Author: Claude Code | Date: 2026-02-16
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiment_data import Experiment, Player, Session
from classify_states_io import (
    OUTPUT_FILE, PROMISE_FILE, load_experiment, load_promise_lookup,
    build_lookup_from_df as _build_lookup_from_df,
    build_group_mean_index as _build_group_mean_index,
    obs_to_row as _obs_to_row,
)

# DEFAULT THRESHOLDS
DEFAULT_GROUP_THRESHOLD = 50.0
DEFAULT_PLAYER_THRESHOLD = 12.5


def main():
    """Main execution flow for state classification."""
    experiment = load_experiment()
    promise_lookup = load_promise_lookup(PROMISE_FILE)
    classification = build_state_classification(experiment, promise_lookup)
    classification.to_csv(OUTPUT_FILE)
    print(f"Wrote {OUTPUT_FILE}")
    print(classification.summary())


@dataclass
class Observation:
    """Single player-round observation within a group state."""

    session_code: str
    treatment: int
    segment: str
    round_num: int
    group_id: int
    label: str
    contribution: float
    made_promise: bool
    player: Player


class PlayerHistory:
    """Full trajectory for one player across the entire session."""

    def __init__(self, session_code: str, treatment: int, label: str,
                 rounds_data: Dict[str, Dict[int, Player]]):
        self.session_code = session_code
        self.treatment = treatment
        self.label = label
        self.key = f"{session_code}:{label}"
        self.rounds_data = rounds_data
        self.observations: List[Observation] = []

    @property
    def mean_contribution(self) -> float:
        """Mean contribution across all observations."""
        if not self.observations:
            return 0.0
        return sum(o.contribution for o in self.observations) / len(self.observations)

    @property
    def total_rounds(self) -> int:
        """Total rounds with data across all segments."""
        return sum(len(rounds) for rounds in self.rounds_data.values())


class MatrixCell:
    """One cell of the 2x2 matrix (behavior x promise)."""

    def __init__(self, behavior_label: str, promise_label: str):
        self.behavior_label = behavior_label
        self.promise_label = promise_label
        self.players: Dict[str, PlayerHistory] = {}

    def add_observation(self, obs: Observation, session: Session):
        """Add an observation, creating PlayerHistory if needed."""
        key = f"{obs.session_code}:{obs.label}"
        if key not in self.players:
            rounds_data = session.get_player_across_session(obs.label)
            self.players[key] = PlayerHistory(
                obs.session_code, obs.treatment, obs.label, rounds_data
            )
        self.players[key].observations.append(obs)

    @property
    def count(self) -> int:
        """Number of unique players in this cell."""
        return len(self.players)

    @property
    def observation_count(self) -> int:
        """Total observations across all players in this cell."""
        return sum(len(ph.observations) for ph in self.players.values())

    @property
    def mean_contribution(self) -> float:
        """Mean contribution across all observations in this cell."""
        all_obs = [o for ph in self.players.values() for o in ph.observations]
        if not all_obs:
            return 0.0
        return sum(o.contribution for o in all_obs) / len(all_obs)


class TwoByTwoMatrix:
    """2x2 matrix of behavior (cooperative/noncooperative) x promise axis."""

    def __init__(self):
        self.cooperative_promise = MatrixCell("cooperative", "promise")
        self.cooperative_no_promise = MatrixCell("cooperative", "no_promise")
        self.noncooperative_promise = MatrixCell("noncooperative", "promise")
        self.noncooperative_no_promise = MatrixCell("noncooperative", "no_promise")

    def __getitem__(self, key: Tuple[str, str]) -> MatrixCell:
        """Access cell by (behavior, promise) tuple."""
        behavior, promise = key
        return self._cell_map()[(behavior, promise)]

    def _cell_map(self) -> Dict[Tuple[str, str], MatrixCell]:
        """Map (behavior, promise) tuples to cells."""
        return {
            ("cooperative", "promise"): self.cooperative_promise,
            ("cooperative", "no_promise"): self.cooperative_no_promise,
            ("noncooperative", "promise"): self.noncooperative_promise,
            ("noncooperative", "no_promise"): self.noncooperative_no_promise,
        }

    @property
    def cells(self) -> List[MatrixCell]:
        """All four cells as a list."""
        return [
            self.cooperative_promise,
            self.cooperative_no_promise,
            self.noncooperative_promise,
            self.noncooperative_no_promise,
        ]


@dataclass
class GroupObservation:
    """A single group-round classified as cooperative or noncooperative."""

    session_code: str
    segment: str
    round_num: int
    group_id: int
    mean_contribution: float
    is_cooperative: bool


class CooperativeState:
    """One side of the cooperative/noncooperative classification."""

    def __init__(self, label: str):
        self.label = label
        self.matrix = TwoByTwoMatrix()
        self.group_observations: List[GroupObservation] = []

    @property
    def group_count(self) -> int:
        """Number of group-round observations in this state."""
        return len(self.group_observations)

    @property
    def observation_count(self) -> int:
        """Total player observations across all matrix cells."""
        return sum(cell.observation_count for cell in self.matrix.cells)


class StateClassification:
    """Top-level container for the full state classification result."""

    def __init__(self, group_threshold: float, player_threshold: float):
        self.group_threshold = group_threshold
        self.player_threshold = player_threshold
        self.cooperative = CooperativeState("cooperative")
        self.noncooperative = CooperativeState("noncooperative")

    def summary(self) -> str:
        """Human-readable summary of classification results."""
        sep = "=" * 50
        lines = [sep, "STATE CLASSIFICATION SUMMARY", sep,
                 f"Group threshold: {self.group_threshold}%",
                 f"Player threshold: {self.player_threshold}"]
        for state in [self.cooperative, self.noncooperative]:
            lines.append(f"\n{state.label.upper()} STATE:")
            lines.append(f"  Group-rounds: {state.group_count}")
            lines.append(f"  Player observations: {state.observation_count}")
            for cell in state.matrix.cells:
                lines.append(f"  {cell.behavior_label}/{cell.promise_label}: "
                             f"{cell.count} players, {cell.observation_count} obs")
        lines.append(sep)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Export classification to a flat DataFrame, one row per player-round."""
        rows = []
        for state in [self.cooperative, self.noncooperative]:
            group_means = _build_group_mean_index(state.group_observations)
            for cell in state.matrix.cells:
                for ph in cell.players.values():
                    for obs in ph.observations:
                        rows.append(_obs_to_row(obs, state, cell, group_means, self))
        return pd.DataFrame(rows)

    def to_csv(self, filepath: Path):
        """Write classification DataFrame to CSV."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(filepath, index=False)


def build_state_classification(
    experiment: Experiment,
    promise_lookup: dict,
    group_threshold: float = DEFAULT_GROUP_THRESHOLD,
    player_threshold: float = DEFAULT_PLAYER_THRESHOLD,
) -> StateClassification:
    """Build state classification from experiment data."""
    classification = StateClassification(group_threshold, player_threshold)
    for session_code, session in experiment.sessions.items():
        _classify_session(
            session_code, session, promise_lookup,
            group_threshold, player_threshold, classification
        )
    return classification


def _classify_session(session_code, session, promise_lookup,
                      group_threshold, player_threshold, classification):
    """Classify all supergame group-rounds in a session."""
    for segment_name, segment in session.segments.items():
        if not segment_name.startswith('supergame'):
            continue
        for round_num in sorted(segment.rounds.keys()):
            round_obj = segment.get_round(round_num)
            for group_id, group in round_obj.groups.items():
                _classify_group_round(
                    session_code, session, segment_name, round_num,
                    group_id, group, promise_lookup,
                    group_threshold, player_threshold, classification
                )


def _group_mean_contribution(group) -> float:
    """Calculate mean contribution for a group."""
    contributions = [p.contribution or 0 for p in group.players.values()]
    return sum(contributions) / len(contributions) if contributions else 0


def _is_group_cooperative(group, threshold) -> tuple:
    """Determine if group is cooperative. Returns (mean_contrib, is_cooperative)."""
    mean_contrib = _group_mean_contribution(group)
    return mean_contrib, (mean_contrib / 25.0) * 100 >= threshold


def _classify_group_round(session_code, session, segment_name, round_num,
                          group_id, group, promise_lookup,
                          group_threshold, player_threshold, cls):
    """Classify a single group-round and its players."""
    mean_contrib, is_coop = _is_group_cooperative(group, group_threshold)
    state = cls.cooperative if is_coop else cls.noncooperative
    group_obs = GroupObservation(session_code, segment_name, round_num, group_id, mean_contrib, is_coop)
    state.group_observations.append(group_obs)
    for label, player in group.players.items():
        _classify_player_in_group(
            session_code, session, segment_name, round_num,
            group_id, label, player, promise_lookup, player_threshold, state)


def _classify_player_in_group(session_code, session, segment_name, round_num,
                              group_id, label, player, promise_lookup,
                              player_threshold, state):
    """Classify a single player within a group-round into a matrix cell."""
    contribution = player.contribution or 0
    made_promise = _get_promise(promise_lookup, session_code, segment_name, round_num, label)
    behavior = "cooperative" if contribution >= player_threshold else "noncooperative"
    promise_axis = "promise" if made_promise else "no_promise"
    obs = Observation(session_code, session.treatment, segment_name, round_num,
                      group_id, label, contribution, made_promise, player)
    state.matrix[(behavior, promise_axis)].add_observation(obs, session)


def _get_promise(lookup, session_code, segment, round_num, label) -> bool:
    """Check if player made a promise. Round 1 always returns False."""
    if round_num == 1:
        return False
    return lookup.get((session_code, segment, round_num, label), False)


# %%
if __name__ == "__main__":
    main()
