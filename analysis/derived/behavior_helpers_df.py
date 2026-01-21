"""
DataFrame-based API for liar/sucker behavioral classifications.

Provides functions for computing behavioral flags from pandas DataFrames
without requiring the Experiment data structure.

Author: Claude Code
Date: 2026-01-17
"""

from typing import Literal

import pandas as pd

# THRESHOLDS
STRICT_THRESHOLD = 20  # Contribution >= 20 honors promise (strict)
LENIENT_THRESHOLD = 5  # Contribution >= 5 honors promise (lenient)
MAX_CONTRIBUTION = 25  # Full contribution required to be a sucker


# =====
# Main entry point
# =====
def main():
    """Demo usage of DataFrame-based helper functions."""
    print("behavior_helpers_df.py - DataFrame API for sucker/liar classification")
    print("Import and use individual functions in your analysis scripts.")


# =====
# Simple threshold functions
# =====
def is_promise_broken_strict(contribution: float) -> bool:
    """Check if contribution breaks promise under strict threshold (< 20)."""
    return contribution < STRICT_THRESHOLD


def is_promise_broken_lenient(contribution: float) -> bool:
    """Check if contribution breaks promise under lenient threshold (< 5)."""
    return contribution < LENIENT_THRESHOLD


# =====
# DataFrame-based liar classification
# =====
def compute_liar_flags(
    df: pd.DataFrame,
    threshold: Literal['strict', 'lenient'] = 'strict'
) -> pd.DataFrame:
    """Compute liar flags for each player-round from a DataFrame.

    A player becomes a liar if they made/broke a promise. Flag persists.
    """
    result = df.copy()
    threshold_func = _get_threshold_func(threshold)
    col_name = f'is_liar_{threshold}'
    result[col_name] = False

    for (session, segment), segment_df in result.groupby(['session_code', 'segment']):
        _compute_liar_for_segment_df(result, segment_df, threshold_func, col_name)

    return result


def _get_threshold_func(threshold: Literal['strict', 'lenient']):
    """Return appropriate threshold function based on type."""
    return is_promise_broken_strict if threshold == 'strict' else is_promise_broken_lenient


def _compute_liar_for_segment_df(
    result: pd.DataFrame, segment_df: pd.DataFrame,
    threshold_func, col_name: str
) -> None:
    """Compute liar flags within a single segment (DataFrame API)."""
    rounds = sorted(segment_df['round'].unique())
    player_is_liar = {}

    for round_num in rounds:
        round_df = segment_df[segment_df['round'] == round_num]
        _process_round_liar_df(result, round_df, round_num, player_is_liar, threshold_func, col_name)


def _process_round_liar_df(result, round_df, round_num, player_is_liar, threshold_func, col_name):
    """Process a single round for liar flag computation."""
    for _, row in round_df.iterrows():
        idx, label = row.name, row['label']
        result.loc[idx, col_name] = False if round_num == 1 else player_is_liar.get(label, False)
        _check_and_flag_liar_df(row, label, player_is_liar, threshold_func)


def _check_and_flag_liar_df(row, label, player_is_liar, threshold_func) -> None:
    """Check if player broke promise and flag them as liar (DataFrame API)."""
    if row['made_promise'] and threshold_func(row['contribution']):
        player_is_liar[label] = True


# =====
# DataFrame-based sucker classification
# =====
def compute_sucker_flags(
    df: pd.DataFrame,
    threshold: Literal['strict', 'lenient'] = 'strict'
) -> pd.DataFrame:
    """Compute sucker flags for each player-round from a DataFrame.

    A player becomes a sucker if they contributed max when group member broke promise.
    """
    result = df.copy()
    threshold_func = _get_threshold_func(threshold)
    col_name = f'is_sucker_{threshold}'
    result[col_name] = False

    for (session, segment), segment_df in result.groupby(['session_code', 'segment']):
        _compute_sucker_for_segment_df(result, segment_df, threshold_func, col_name)

    return result


def _compute_sucker_for_segment_df(
    result: pd.DataFrame, segment_df: pd.DataFrame,
    threshold_func, col_name: str
) -> None:
    """Compute sucker flags within a single segment (DataFrame API)."""
    rounds = sorted(segment_df['round'].unique())
    player_is_sucker = {}

    for round_num in rounds:
        round_df = segment_df[segment_df['round'] == round_num]
        _process_round_sucker_df(
            result, round_df, round_num, threshold_func, player_is_sucker, col_name
        )


def _process_round_sucker_df(
    result, round_df, round_num, threshold_func, player_is_sucker, col_name
):
    """Process a single round for sucker classification (DataFrame API)."""
    # Find groups with promise-breakers in THIS round only
    groups_with_liar_this_round = _find_groups_with_liar(round_df, threshold_func)

    # Set flags and check for new suckering events
    for _, row in round_df.iterrows():
        idx = row.name
        label = row['label']
        group = row['group']

        if round_num == 1:
            result.loc[idx, col_name] = False
        elif player_is_sucker.get(label, False):
            result.loc[idx, col_name] = True
        else:
            result.loc[idx, col_name] = False

        # Check if suckered THIS round (contribution 25 + groupmate broke promise)
        if group in groups_with_liar_this_round and row['contribution'] == MAX_CONTRIBUTION:
            player_is_sucker[label] = True


def _find_groups_with_liar(round_df, threshold_func) -> set:
    """Find groups that have a promise-breaker in this specific round."""
    groups = set()
    for _, row in round_df.iterrows():
        if row['made_promise'] and threshold_func(row['contribution']):
            groups.add(row['group'])
    return groups


# =====
# Combined classification (DataFrame API)
# =====
def classify_player_behavior(df: pd.DataFrame) -> pd.DataFrame:
    """Classify player behavior with all liar and sucker flags.

    Adds: is_liar_strict, is_liar_lenient, is_sucker_strict, is_sucker_lenient.
    """
    result = compute_liar_flags(df, threshold='strict')
    result = compute_liar_flags(result, threshold='lenient')
    result = compute_sucker_flags(result, threshold='strict')
    result = compute_sucker_flags(result, threshold='lenient')
    return result


# %%
if __name__ == "__main__":
    main()
