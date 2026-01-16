"""
Tests for group contribution calculations.

Verifies that:
1. Raw CSV data has correct group totals (sum of player contributions)
2. Raw CSV individual_share = round(total_contribution * MPCR)
3. Loaded Session data preserves raw CSV values

Author: Test Infrastructure
Date: 2026-01-15
"""

import pytest
import pandas as pd
from experiment_data import Session

# CONSTANTS
MPCR = 0.4  # Marginal per capita return (multiplier for public goods game)


# =====
# Test: Group total in raw CSV equals sum of player contributions
# =====
def test_group_total_contribution_equals_sum(t1_raw_df: pd.DataFrame):
    """Raw CSV total_contribution should equal sum of player contributions."""
    # Test across multiple supergames and rounds
    for sg in range(1, 6):
        for rd in range(1, 5):
            total_col = f'supergame{sg}.{rd}.group.total_contribution'
            contrib_col = f'supergame{sg}.{rd}.player.contribution'
            group_id_col = f'supergame{sg}.{rd}.group.id_in_subsession'

            if total_col not in t1_raw_df.columns:
                continue

            for group_id in t1_raw_df[group_id_col].dropna().unique():
                mask = t1_raw_df[group_id_col] == group_id
                player_sum = t1_raw_df.loc[mask, contrib_col].sum()
                raw_total = t1_raw_df.loc[mask, total_col].iloc[0]

                assert raw_total == pytest.approx(player_sum), (
                    f"Raw CSV mismatch in SG{sg}R{rd}G{int(group_id)}: "
                    f"total_contribution={raw_total}, sum={player_sum}"
                )


# =====
# Test: Loaded group total matches raw CSV value
# =====
def test_group_total_contribution_matches_raw(
    loaded_t1_session: Session,
    t1_raw_df: pd.DataFrame
):
    """Loaded Group.total_contribution should match the raw CSV value."""
    for segment_name, segment in loaded_t1_session.segments.items():
        if not segment_name.startswith('supergame'):
            continue

        for round_num, round_obj in segment.rounds.items():
            col_name = f'{segment_name}.{round_num}.group.total_contribution'
            group_id_col = f'{segment_name}.{round_num}.group.id_in_subsession'

            if col_name not in t1_raw_df.columns:
                continue

            for group_id, group in round_obj.groups.items():
                # Get raw value for this group
                mask = t1_raw_df[group_id_col] == group_id
                raw_values = t1_raw_df.loc[mask, col_name].dropna()

                if raw_values.empty:
                    continue

                raw_total = raw_values.iloc[0]

                assert group.total_contribution == pytest.approx(raw_total), (
                    f"Loaded vs raw mismatch in {segment_name} R{round_num} G{group_id}: "
                    f"loaded={group.total_contribution}, raw={raw_total}"
                )


# =====
# Test: Loaded player contributions sum to group total
# =====
def test_loaded_player_sum_equals_group_total(loaded_t1_session: Session):
    """Loaded player contributions should sum to group total_contribution."""
    for segment_name, segment in loaded_t1_session.segments.items():
        if not segment_name.startswith('supergame'):
            continue

        for round_num, round_obj in segment.rounds.items():
            for group_id, group in round_obj.groups.items():
                player_contributions = [
                    p.contribution for p in group.players.values()
                    if p.contribution is not None
                ]

                if not player_contributions:
                    continue

                expected_total = sum(player_contributions)

                assert group.total_contribution == pytest.approx(expected_total), (
                    f"Sum mismatch in {segment_name} R{round_num} G{group_id}: "
                    f"total={group.total_contribution}, sum={expected_total}"
                )


# =====
# Test: Individual share calculation (uses rounding)
# =====
def test_individual_share_calculation(t1_raw_df: pd.DataFrame):
    """Verify raw CSV individual_share = round(total_contribution * MPCR)."""
    for sg in range(1, 6):
        for rd in range(1, 5):
            total_col = f'supergame{sg}.{rd}.group.total_contribution'
            share_col = f'supergame{sg}.{rd}.group.individual_share'
            group_id_col = f'supergame{sg}.{rd}.group.id_in_subsession'

            if total_col not in t1_raw_df.columns:
                continue

            for group_id in t1_raw_df[group_id_col].dropna().unique():
                mask = t1_raw_df[group_id_col] == group_id
                total = t1_raw_df.loc[mask, total_col].iloc[0]
                share = t1_raw_df.loc[mask, share_col].iloc[0]

                # oTree rounds the individual share calculation
                expected_share = round(total * MPCR)

                assert share == pytest.approx(expected_share), (
                    f"Share mismatch in SG{sg}R{rd}G{int(group_id)}: "
                    f"share={share}, expected={expected_share}"
                )


# =====
# Test: T2 raw CSV verification
# =====
def test_group_contribution_sample_t2(t2_raw_df: pd.DataFrame):
    """Verify T2 raw CSV group contributions and individual shares."""
    # Filter to valid participants only (exclude NaN labels)
    valid_df = t2_raw_df[t2_raw_df['participant.label'].notna()].copy()

    # Test 1: Raw CSV totals match player sums
    for sg in range(1, 6):
        for rd in range(1, 5):
            total_col = f'supergame{sg}.{rd}.group.total_contribution'
            contrib_col = f'supergame{sg}.{rd}.player.contribution'
            group_id_col = f'supergame{sg}.{rd}.group.id_in_subsession'

            if total_col not in valid_df.columns:
                continue

            for group_id in valid_df[group_id_col].dropna().unique():
                mask = valid_df[group_id_col] == group_id
                player_sum = valid_df.loc[mask, contrib_col].sum()
                raw_total = valid_df.loc[mask, total_col].iloc[0]

                assert raw_total == pytest.approx(player_sum), (
                    f"T2 raw mismatch in SG{sg}R{rd}G{int(group_id)}"
                )

    # Test 2: Individual share uses rounding
    for sg in range(1, 6):
        for rd in range(1, 5):
            total_col = f'supergame{sg}.{rd}.group.total_contribution'
            share_col = f'supergame{sg}.{rd}.group.individual_share'
            group_id_col = f'supergame{sg}.{rd}.group.id_in_subsession'

            if total_col not in valid_df.columns:
                continue

            for group_id in valid_df[group_id_col].dropna().unique():
                mask = valid_df[group_id_col] == group_id
                total = valid_df.loc[mask, total_col].iloc[0]
                share = valid_df.loc[mask, share_col].iloc[0]

                expected_share = round(total * MPCR)

                assert share == pytest.approx(expected_share), (
                    f"T2 share mismatch in SG{sg}R{rd}G{int(group_id)}"
                )


# =====
# Test: T2 loaded data matches raw
# =====
def test_t2_loaded_matches_raw(
    loaded_t2_session: Session,
    t2_raw_df: pd.DataFrame
):
    """Verify T2 loaded data matches raw CSV values."""
    valid_df = t2_raw_df[t2_raw_df['participant.label'].notna()].copy()

    for segment_name, segment in loaded_t2_session.segments.items():
        if not segment_name.startswith('supergame'):
            continue

        for round_num, round_obj in segment.rounds.items():
            col_name = f'{segment_name}.{round_num}.group.total_contribution'
            group_id_col = f'{segment_name}.{round_num}.group.id_in_subsession'

            if col_name not in valid_df.columns:
                continue

            for group_id, group in round_obj.groups.items():
                mask = valid_df[group_id_col] == group_id
                raw_values = valid_df.loc[mask, col_name].dropna()

                if raw_values.empty:
                    continue

                raw_total = raw_values.iloc[0]

                assert group.total_contribution == pytest.approx(raw_total), (
                    f"T2 loaded vs raw mismatch in {segment_name} R{round_num} G{group_id}"
                )
