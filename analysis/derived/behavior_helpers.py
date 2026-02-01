"""
Helper functions for liar/sucker behavioral classifications.
For DataFrame-based operations, see behavior_helpers_df.py.
Author: Claude Code | Date: 2026-01-17
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiment_data import Experiment

# Re-export DataFrame API functions for backwards compatibility
from behavior_helpers_df import (
    is_promise_broken_20,
    is_promise_broken_5,
    compute_liar_flags,
    compute_sucker_flags,
    classify_player_behavior,
    THRESHOLD_20,
    THRESHOLD_5,
    MAX_CONTRIBUTION,
)

# FILE PATHS
DEFAULT_PROMISE_FILE = Path(__file__).parent.parent / 'datastore' / 'derived' / 'promise_classifications.csv'


# =====
# Main entry point
# =====
def main():
    """Demo usage of helper functions."""
    print("behavior_helpers.py - Helper functions for sucker/liar classification")
    print("Import and use individual functions in your analysis scripts.")


# =====
# Data loading
# =====
def load_promise_data(filepath: Path = DEFAULT_PROMISE_FILE) -> pd.DataFrame:
    """Load and parse promise_classifications.csv.

    Args:
        filepath: Path to the promise classifications CSV file.

    Returns:
        DataFrame with promise classification data.
    """
    return pd.read_csv(filepath)


# =====
# Player-round record building
# =====
def build_all_player_rounds(experiment: Experiment) -> List[Dict]:
    """Create records for ALL player-rounds (not just chatters).

    Iterates through all sessions, segments, rounds, and groups to build
    a complete list of player-round records.

    Returns:
        List of dicts with keys: session_code, treatment, segment, round,
        group, label, participant_id, contribution, payoff.
    """
    records = []
    for session_code, session in experiment.sessions.items():
        for segment_name, segment in session.segments.items():
            if not segment_name.startswith('supergame'):
                continue
            records.extend(_build_segment_records(session_code, session.treatment,
                                                   segment_name, segment))
    return records


def _build_segment_records(session_code: str, treatment: int,
                           segment_name: str, segment) -> List[Dict]:
    """Build player records for a single segment."""
    records = []
    for round_num, round_obj in segment.rounds.items():
        for group_id, group in round_obj.groups.items():
            for label, player in group.players.items():
                records.append(_build_player_record(
                    session_code, treatment, segment_name, round_num,
                    group_id, label, player
                ))
    return records


def _build_player_record(session_code: str, treatment: int, segment: str,
                         round_num: int, group_id: int, label: str,
                         player) -> Dict:
    """Build a single player-round record dictionary."""
    return {
        'session_code': session_code,
        'treatment': treatment,
        'segment': segment,
        'round': round_num,
        'group': group_id,
        'label': label,
        'participant_id': player.participant_id,
        'contribution': player.contribution,
        'payoff': player.payoff,
    }


# =====
# Group membership
# =====
def get_group_members(experiment: Experiment, session_code: str,
                      segment: str, round_num: int, label: str) -> List[str]:
    """Get list of group member labels for a given player in a given round.

    Returns:
        List of player labels in the same group (excluding the target player).
    """
    session = experiment.get_session(session_code)
    if not session:
        return []

    segment_obj = session.get_segment(segment)
    if not segment_obj:
        return []

    round_obj = segment_obj.get_round(round_num)
    if not round_obj:
        return []

    return _find_group_members(round_obj, label)


def _find_group_members(round_obj, target_label: str) -> List[str]:
    """Find all group members for a target player in a round."""
    for group in round_obj.groups.values():
        if target_label in group.players:
            return [lbl for lbl in group.players.keys() if lbl != target_label]
    return []


# =====
# Promise checking
# =====
def check_promise_made(promise_df: pd.DataFrame, session_code: str,
                       segment: str, round_num: int, label: str) -> bool:
    """Check if player made a promise in a specific round.

    Looks up the player in the promise dataframe and checks if they
    had any promise classifications (promise_count > 0).
    """
    mask = (
        (promise_df['session_code'] == session_code) &
        (promise_df['segment'] == segment) &
        (promise_df['round'] == round_num) &
        (promise_df['label'] == label)
    )
    matching = promise_df.loc[mask, 'promise_count']
    if matching.empty:
        return False
    return matching.iloc[0] > 0


def is_broken_promise(contribution: float, threshold: float) -> bool:
    """Check if contribution is below the threshold (broken promise).

    A promise is considered broken when actual contribution falls below
    the expected cooperation threshold.
    """
    if contribution is None:
        return False
    return contribution < threshold


# =====
# Liar flag computation (Experiment API)
# =====
def compute_liar_flag(promise_df: pd.DataFrame, experiment: Experiment,
                      session_code: str, segment: str, current_round: int,
                      label: str, threshold: float) -> bool:
    """Compute whether a player is flagged as a liar.

    Returns False if round == 1. Returns True if liar condition ever met.
    """
    if current_round == 1:
        return False
    return _check_any_prior_round_liar(
        promise_df, experiment, session_code, segment, current_round, label, threshold
    )


def _check_any_prior_round_liar(promise_df, experiment, session_code, segment,
                                 current_round, label, threshold) -> bool:
    """Check if player lied in any prior round."""
    for prior_round in range(1, current_round):
        if _check_liar_in_round(promise_df, experiment, session_code,
                                segment, prior_round, label, threshold):
            return True
    return False


def _check_liar_in_round(promise_df: pd.DataFrame, experiment: Experiment,
                         session_code: str, segment: str, round_num: int,
                         label: str, threshold: float) -> bool:
    """Check if player lied in a specific round (promise made + broken)."""
    if not check_promise_made(promise_df, session_code, segment, round_num, label):
        return False

    contribution = _get_player_contribution(experiment, session_code,
                                            segment, round_num, label)
    return is_broken_promise(contribution, threshold)


def _get_player_contribution(experiment: Experiment, session_code: str,
                             segment: str, round_num: int,
                             label: str) -> Optional[float]:
    """Get a player's contribution for a specific round."""
    session = experiment.get_session(session_code)
    if not session:
        return None

    segment_obj = session.get_segment(segment)
    if not segment_obj:
        return None

    round_obj = segment_obj.get_round(round_num)
    if not round_obj:
        return None

    player = round_obj.get_player(label)
    return player.contribution if player else None


# =====
# Sucker flag computation (Experiment API)
# =====
def compute_sucker_flag(promise_df: pd.DataFrame, experiment: Experiment,
                        session_code: str, segment: str, current_round: int,
                        label: str, threshold: float) -> bool:
    """Compute whether a player is flagged as a sucker.

    Returns False if round == 1. Returns True if sucker condition ever met.
    """
    if current_round == 1:
        return False
    return _check_any_prior_round_sucker(
        promise_df, experiment, session_code, segment, current_round, label, threshold
    )


def _check_any_prior_round_sucker(promise_df, experiment, session_code, segment,
                                   current_round, label, threshold) -> bool:
    """Check if player was suckered in any prior round."""
    for prior_round in range(1, current_round):
        if _check_sucker_in_round(promise_df, experiment, session_code,
                                  segment, prior_round, label, threshold):
            return True
    return False


def _check_sucker_in_round(promise_df: pd.DataFrame, experiment: Experiment,
                           session_code: str, segment: str, round_num: int,
                           label: str, threshold: float) -> bool:
    """Check if player was suckered in a specific round.

    Sucker condition: player contributed 25 AND a group member broke a promise.
    """
    contribution = _get_player_contribution(experiment, session_code,
                                            segment, round_num, label)
    if contribution != 25:
        return False

    group_members = get_group_members(experiment, session_code,
                                      segment, round_num, label)
    return _any_member_broke_promise(promise_df, experiment, session_code,
                                     segment, round_num, group_members, threshold)


def _any_member_broke_promise(promise_df: pd.DataFrame, experiment: Experiment,
                              session_code: str, segment: str, round_num: int,
                              members: List[str], threshold: float) -> bool:
    """Check if any group member made a promise but contributed below threshold."""
    for member_label in members:
        if not check_promise_made(promise_df, session_code, segment,
                                  round_num, member_label):
            continue
        member_contrib = _get_player_contribution(experiment, session_code,
                                                  segment, round_num, member_label)
        if is_broken_promise(member_contrib, threshold):
            return True
    return False


# %%
if __name__ == "__main__":
    main()
