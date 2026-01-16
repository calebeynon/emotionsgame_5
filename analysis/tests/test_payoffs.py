"""
Tests for payoff accuracy in experiment_data module.

Verifies that Player.payoff values match raw CSV data and that
the payoff formula (25 - contribution + individual_share) is correct.

Author: Test Infrastructure
Date: 2026-01-15
"""

import pytest
import pandas as pd


# =====
# Main test functions
# =====
def test_player_payoff_matches_raw_csv(loaded_t1_session, t1_raw_df):
    """Verify Player.payoff matches the raw CSV column value."""
    supergame = loaded_t1_session.get_supergame(1)
    round_obj = supergame.get_round(1)

    # Build lookup from raw CSV for supergame1.1 payoffs by participant label
    raw_payoffs = _build_raw_payoff_lookup(t1_raw_df, 'supergame1.1')

    # Verify each player's payoff matches raw CSV
    for label, player in round_obj.players.items():
        if label in raw_payoffs:
            assert player.payoff == pytest.approx(raw_payoffs[label]), \
                f"Player {label} payoff mismatch: got {player.payoff}, expected {raw_payoffs[label]}"


def test_payoff_calculation_formula(loaded_t1_session, t1_raw_df):
    """Verify payoff = (25 - contribution) + individual_share using raw CSV values."""
    supergame = loaded_t1_session.get_supergame(1)
    round_obj = supergame.get_round(1)

    # Build lookup for individual_share by player label from raw CSV
    raw_shares_by_label = _build_raw_share_by_label_lookup(t1_raw_df, 'supergame1.1')

    for label, player in round_obj.players.items():
        if player.contribution is not None and player.payoff is not None:
            if label in raw_shares_by_label:
                individual_share = raw_shares_by_label[label]
                expected_payoff = (25 - player.contribution) + individual_share
                assert player.payoff == pytest.approx(expected_payoff), \
                    f"Player {label} formula mismatch: (25 - {player.contribution}) + {individual_share} = {expected_payoff}, got {player.payoff}"


def test_individual_share_matches_raw(loaded_t1_session, t1_raw_df):
    """Verify individual_share values are correctly present in raw CSV data."""
    supergame = loaded_t1_session.get_supergame(1)
    round_obj = supergame.get_round(1)

    # Build lookup for individual_share by player label from raw CSV
    raw_shares_by_label = _build_raw_share_by_label_lookup(t1_raw_df, 'supergame1.1')

    # Verify each player has an individual_share in the raw data
    for label, player in round_obj.players.items():
        assert label in raw_shares_by_label, \
            f"Player {label} missing individual_share in raw CSV"
        assert raw_shares_by_label[label] > 0, \
            f"Player {label} has invalid individual_share: {raw_shares_by_label[label]}"


def test_payoff_sample_t2(loaded_t2_session, t2_raw_df):
    """Same payoff verification for t2 session."""
    supergame = loaded_t2_session.get_supergame(1)
    round_obj = supergame.get_round(1)

    # Build lookups from raw CSV
    raw_payoffs = _build_raw_payoff_lookup(t2_raw_df, 'supergame1.1')
    raw_shares_by_label = _build_raw_share_by_label_lookup(t2_raw_df, 'supergame1.1')

    # Verify payoffs match raw CSV
    for label, player in round_obj.players.items():
        if label in raw_payoffs:
            assert player.payoff == pytest.approx(raw_payoffs[label]), \
                f"T2 Player {label} payoff mismatch: got {player.payoff}, expected {raw_payoffs[label]}"

    # Verify payoff formula using raw CSV individual_share values
    for label, player in round_obj.players.items():
        if player.contribution is not None and player.payoff is not None:
            if label in raw_shares_by_label:
                individual_share = raw_shares_by_label[label]
                expected = (25 - player.contribution) + individual_share
                assert player.payoff == pytest.approx(expected), \
                    f"T2 Player {label} formula mismatch: (25 - {player.contribution}) + {individual_share} = {expected}, got {player.payoff}"


# =====
# Helper functions
# =====
def _build_raw_payoff_lookup(df, segment_round_prefix):
    """Build lookup dict of participant.label -> payoff from raw CSV."""
    payoff_col = f'{segment_round_prefix}.player.payoff'
    label_col = 'participant.label'

    payoffs = {}
    for _, row in df.iterrows():
        label = row[label_col]
        payoff = row[payoff_col]
        if pd.notna(label) and pd.notna(payoff):
            payoffs[label] = float(payoff)

    return payoffs


def _build_raw_share_by_label_lookup(df, segment_round_prefix):
    """Build lookup dict of participant.label -> individual_share from raw CSV."""
    share_col = f'{segment_round_prefix}.group.individual_share'
    label_col = 'participant.label'

    shares = {}
    for _, row in df.iterrows():
        label = row[label_col]
        share = row[share_col]
        if pd.notna(label) and pd.notna(share):
            shares[label] = float(share)

    return shares
