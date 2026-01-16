"""
Tests for contribution data accuracy in experiment_data module.

Verifies that Player.contribution values match raw CSV data and are within valid ranges.

Author: Test Infrastructure
Date: 2026-01-15
"""

import pytest
import pandas as pd
from experiment_data import Session


# =====
# Constants
# =====
ENDOWMENT = 25  # Maximum contribution value


# =====
# Helper functions
# =====
def get_raw_contribution(raw_df: pd.DataFrame, participant_label: str,
                         supergame: int, round_num: int) -> float:
    """Extract contribution from raw CSV for a specific player/supergame/round."""
    col_name = f"supergame{supergame}.{round_num}.player.contribution"
    row = raw_df[raw_df['participant.label'] == participant_label]

    if row.empty or col_name not in raw_df.columns:
        return None

    value = row[col_name].iloc[0]
    return float(value) if pd.notna(value) else None


def verify_contributions_match(session: Session, raw_df: pd.DataFrame) -> list:
    """
    Verify all contributions in session match raw CSV values.

    Returns list of mismatches for assertion reporting.
    """
    mismatches = []

    for sg_num in range(1, 6):
        supergame = session.get_supergame(sg_num)
        if not supergame:
            continue

        for round_num, round_obj in supergame.rounds.items():
            for label, player in round_obj.players.items():
                raw_value = get_raw_contribution(raw_df, label, sg_num, round_num)
                loaded_value = player.contribution

                # Both should be None or equal
                if raw_value is None and loaded_value is None:
                    continue
                if raw_value != loaded_value:
                    mismatches.append({
                        'supergame': sg_num,
                        'round': round_num,
                        'player': label,
                        'raw': raw_value,
                        'loaded': loaded_value
                    })

    return mismatches


# =====
# Tests for T1 session
# =====
def test_player_contribution_matches_raw_csv(loaded_t1_session: Session,
                                              t1_raw_df: pd.DataFrame):
    """Verify Player.contribution matches corresponding column in raw CSV."""
    mismatches = verify_contributions_match(loaded_t1_session, t1_raw_df)

    assert len(mismatches) == 0, (
        f"Found {len(mismatches)} contribution mismatches:\n"
        + "\n".join(
            f"  SG{m['supergame']} R{m['round']} {m['player']}: "
            f"raw={m['raw']} loaded={m['loaded']}"
            for m in mismatches[:10]
        )
    )


def test_all_contributions_non_negative(loaded_t1_session: Session):
    """Verify no negative contributions exist."""
    negative_contributions = []

    for sg_num in range(1, 6):
        supergame = loaded_t1_session.get_supergame(sg_num)
        if not supergame:
            continue

        for round_num, round_obj in supergame.rounds.items():
            for label, player in round_obj.players.items():
                if player.contribution is not None and player.contribution < 0:
                    negative_contributions.append({
                        'supergame': sg_num,
                        'round': round_num,
                        'player': label,
                        'contribution': player.contribution
                    })

    assert len(negative_contributions) == 0, (
        f"Found {len(negative_contributions)} negative contributions:\n"
        + "\n".join(
            f"  SG{n['supergame']} R{n['round']} {n['player']}: {n['contribution']}"
            for n in negative_contributions
        )
    )


def test_contribution_values_in_valid_range(loaded_t1_session: Session):
    """Verify all contributions are between 0 and ENDOWMENT (25)."""
    out_of_range = []

    for sg_num in range(1, 6):
        supergame = loaded_t1_session.get_supergame(sg_num)
        if not supergame:
            continue

        for round_num, round_obj in supergame.rounds.items():
            for label, player in round_obj.players.items():
                contrib = player.contribution
                if contrib is not None and (contrib < 0 or contrib > ENDOWMENT):
                    out_of_range.append({
                        'supergame': sg_num,
                        'round': round_num,
                        'player': label,
                        'contribution': contrib
                    })

    assert len(out_of_range) == 0, (
        f"Found {len(out_of_range)} contributions outside valid range [0, {ENDOWMENT}]:\n"
        + "\n".join(
            f"  SG{o['supergame']} R{o['round']} {o['player']}: {o['contribution']}"
            for o in out_of_range
        )
    )


# =====
# Tests for T2 session
# =====
def test_contribution_sample_t2(loaded_t2_session: Session, t2_raw_df: pd.DataFrame):
    """Verify contributions match raw CSV for T2 session."""
    mismatches = verify_contributions_match(loaded_t2_session, t2_raw_df)

    assert len(mismatches) == 0, (
        f"Found {len(mismatches)} contribution mismatches in T2:\n"
        + "\n".join(
            f"  SG{m['supergame']} R{m['round']} {m['player']}: "
            f"raw={m['raw']} loaded={m['loaded']}"
            for m in mismatches[:10]
        )
    )
