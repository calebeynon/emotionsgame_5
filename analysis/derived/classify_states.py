"""
State classification: cooperative/noncooperative group states with promise axis.
Author: Claude Code | Date: 2026-02-16
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_data import Experiment, Player, Session, load_experiment_data

# FILE PATHS
DATA_DIR = Path(__file__).parent.parent / 'datastore'
RAW_DIR = DATA_DIR / 'raw'
PROMISE_FILE = DATA_DIR / 'derived' / 'promise_classifications.csv'

# DEFAULT THRESHOLDS
DEFAULT_GROUP_THRESHOLD = 50.0
DEFAULT_PLAYER_THRESHOLD = 12.5


# =====
# Main function
# =====
def main():
    """Main execution flow for state classification."""
    experiment = load_experiment()
    promise_lookup = load_promise_lookup(PROMISE_FILE)
    classification = build_state_classification(experiment, promise_lookup)
    print(classification.summary())


# =====
# Data classes
# =====
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
        lines = [
            "=" * 50,
            "STATE CLASSIFICATION SUMMARY",
            "=" * 50,
            f"Group threshold: {self.group_threshold}%",
            f"Player threshold: {self.player_threshold}",
        ]
        for state in [self.cooperative, self.noncooperative]:
            lines.append(f"\n{state.label.upper()} STATE:")
            lines.append(f"  Group-rounds: {state.group_count}")
            lines.append(f"  Player observations: {state.observation_count}")
            for cell in state.matrix.cells:
                lines.append(
                    f"  {cell.behavior_label}/{cell.promise_label}: "
                    f"{cell.count} players, {cell.observation_count} obs"
                )
        lines.append("=" * 50)
        return "\n".join(lines)


# =====
# Data loading
# =====
def load_experiment() -> Experiment:
    """Load experiment data from raw session files."""
    return load_experiment_data(build_file_pairs(), name="State Classification")


def build_file_pairs() -> list:
    """Build list of (data_csv, chat_csv, treatment) tuples from raw directory."""
    file_pairs = []
    for data_file in sorted(RAW_DIR.glob("*_data.csv")):
        treatment = extract_treatment(data_file.name)
        chat_file = data_file.with_name(data_file.name.replace("_data", "_chat"))
        chat_path = str(chat_file) if chat_file.exists() else None
        file_pairs.append((str(data_file), chat_path, treatment))
    return file_pairs


def extract_treatment(filename: str) -> int:
    """Extract treatment number from filename like '01_t1_data.csv'."""
    if '_t1_' in filename:
        return 1
    return 2 if '_t2_' in filename else 0


# =====
# Promise lookup
# =====
def load_promise_lookup(filepath: Path) -> dict:
    """Load promise_classifications.csv into lookup dict.

    Returns:
        dict mapping (session, segment, round, label) -> bool
    """
    if not filepath.exists():
        print(f"Warning: Promise file not found at {filepath}")
        return {}
    df = pd.read_csv(filepath)
    return _build_lookup_from_df(df)


def _build_lookup_from_df(df: pd.DataFrame) -> dict:
    """Build lookup dict from promise DataFrame."""
    lookup = {}
    for _, row in df.iterrows():
        key = (row['session_code'], row['segment'], row['round'], row['label'])
        lookup[key] = row['promise_count'] > 0
    return lookup


# =====
# State classification builder
# =====
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
                          group_threshold, player_threshold, classification):
    """Classify a single group-round and its players."""
    mean_contrib, is_coop = _is_group_cooperative(group, group_threshold)
    state = classification.cooperative if is_coop else classification.noncooperative
    state.group_observations.append(GroupObservation(
        session_code, segment_name, round_num, group_id, mean_contrib, is_coop
    ))
    for label, player in group.players.items():
        _classify_player_in_group(
            session_code, session, segment_name, round_num,
            group_id, label, player, promise_lookup, player_threshold, state
        )


def _classify_player_in_group(session_code, session, segment_name, round_num,
                              group_id, label, player, promise_lookup,
                              player_threshold, state):
    """Classify a single player within a group-round into a matrix cell."""
    contribution = player.contribution or 0
    made_promise = _get_promise(promise_lookup, session_code, segment_name, round_num, label)

    behavior = "cooperative" if contribution >= player_threshold else "noncooperative"
    promise_axis = "promise" if made_promise else "no_promise"

    obs = Observation(
        session_code, session.treatment, segment_name, round_num,
        group_id, label, contribution, made_promise, player
    )
    state.matrix[(behavior, promise_axis)].add_observation(obs, session)


def _get_promise(lookup, session_code, segment, round_num, label) -> bool:
    """Check if player made a promise. Round 1 always returns False."""
    if round_num == 1:
        return False
    return lookup.get((session_code, segment, round_num, label), False)


# %%
if __name__ == "__main__":
    main()
