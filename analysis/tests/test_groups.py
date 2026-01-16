"""
Tests for group formation in experiment_data module.

Verifies that groups are correctly formed with:
- 4 players per group
- 4 groups per round
- 16 players per round
- Unique player labels within groups
- Correct group_id assignment

Author: Test Infrastructure
Date: 2026-01-15
"""

import pytest
from experiment_data import Session


# =====
# Group size tests
# =====
def test_four_players_per_group(loaded_t1_session: Session):
    """Every Group has exactly 4 players."""
    for segment_name, segment in loaded_t1_session.segments.items():
        if not segment_name.startswith('supergame'):
            continue

        for round_num, round_obj in segment.rounds.items():
            for group_id, group in round_obj.groups.items():
                assert len(group.players) == 4, (
                    f"{segment_name} round {round_num} group {group_id}: "
                    f"expected 4 players, got {len(group.players)}"
                )


def test_four_groups_per_round(loaded_t1_session: Session):
    """Every Round has exactly 4 groups."""
    for segment_name, segment in loaded_t1_session.segments.items():
        if not segment_name.startswith('supergame'):
            continue

        for round_num, round_obj in segment.rounds.items():
            assert len(round_obj.groups) == 4, (
                f"{segment_name} round {round_num}: "
                f"expected 4 groups, got {len(round_obj.groups)}"
            )


def test_sixteen_players_per_round(loaded_t1_session: Session):
    """Every Round has 16 players total."""
    for segment_name, segment in loaded_t1_session.segments.items():
        if not segment_name.startswith('supergame'):
            continue

        for round_num, round_obj in segment.rounds.items():
            assert len(round_obj.players) == 16, (
                f"{segment_name} round {round_num}: "
                f"expected 16 players, got {len(round_obj.players)}"
            )


# =====
# Player label uniqueness tests
# =====
def test_player_labels_unique_in_group(loaded_t1_session: Session):
    """No duplicate player labels within the same group."""
    for segment_name, segment in loaded_t1_session.segments.items():
        if not segment_name.startswith('supergame'):
            continue

        for round_num, round_obj in segment.rounds.items():
            for group_id, group in round_obj.groups.items():
                labels = list(group.players.keys())
                unique_labels = set(labels)

                assert len(labels) == len(unique_labels), (
                    f"{segment_name} round {round_num} group {group_id}: "
                    f"duplicate labels found: {labels}"
                )


# =====
# Group ID consistency tests
# =====
def test_player_group_id_matches_group(loaded_t1_session: Session):
    """Player.group_id matches parent Group.group_id."""
    for segment_name, segment in loaded_t1_session.segments.items():
        if not segment_name.startswith('supergame'):
            continue

        for round_num, round_obj in segment.rounds.items():
            for group_id, group in round_obj.groups.items():
                for label, player in group.players.items():
                    assert player.group_id == group.group_id, (
                        f"{segment_name} round {round_num} player {label}: "
                        f"player.group_id={player.group_id} != "
                        f"group.group_id={group.group_id}"
                    )


def test_group_id_in_subsession_matches_raw(loaded_t1_session: Session, t1_raw_df):
    """Group.group_id matches CSV column group.id_in_subsession."""
    # Get group IDs from raw CSV for supergame1, round 1
    raw_group_col = 'supergame1.1.group.id_in_subsession'

    if raw_group_col not in t1_raw_df.columns:
        pytest.skip(f"Column {raw_group_col} not found in raw data")

    raw_group_ids = set(t1_raw_df[raw_group_col].dropna().astype(int).unique())

    # Get group IDs from loaded session
    supergame1 = loaded_t1_session.get_supergame(1)
    assert supergame1 is not None, "supergame1 not found in session"

    round1 = supergame1.get_round(1)
    assert round1 is not None, "Round 1 not found in supergame1"

    loaded_group_ids = set(round1.groups.keys())

    assert raw_group_ids == loaded_group_ids, (
        f"Group IDs mismatch: raw={raw_group_ids}, loaded={loaded_group_ids}"
    )


# =====
# T2 session tests
# =====
def test_group_formation_t2(loaded_t2_session: Session, t2_raw_df):
    """Same verification for t2: group structure and raw CSV match."""
    # Verify 4 players per group
    for segment_name, segment in loaded_t2_session.segments.items():
        if not segment_name.startswith('supergame'):
            continue

        for round_num, round_obj in segment.rounds.items():
            # 4 groups per round
            assert len(round_obj.groups) == 4, (
                f"T2 {segment_name} round {round_num}: "
                f"expected 4 groups, got {len(round_obj.groups)}"
            )

            # 16 players per round
            assert len(round_obj.players) == 16, (
                f"T2 {segment_name} round {round_num}: "
                f"expected 16 players, got {len(round_obj.players)}"
            )

            for group_id, group in round_obj.groups.items():
                # 4 players per group
                assert len(group.players) == 4, (
                    f"T2 {segment_name} round {round_num} group {group_id}: "
                    f"expected 4 players, got {len(group.players)}"
                )

                # Player group_id matches group
                for label, player in group.players.items():
                    assert player.group_id == group.group_id, (
                        f"T2 {segment_name} round {round_num} player {label}: "
                        f"group_id mismatch"
                    )

    # Verify group IDs match raw CSV
    raw_group_col = 'supergame1.1.group.id_in_subsession'

    if raw_group_col not in t2_raw_df.columns:
        pytest.skip(f"Column {raw_group_col} not found in T2 raw data")

    raw_group_ids = set(t2_raw_df[raw_group_col].dropna().astype(int).unique())

    supergame1 = loaded_t2_session.get_supergame(1)
    assert supergame1 is not None, "T2 supergame1 not found"

    round1 = supergame1.get_round(1)
    assert round1 is not None, "T2 Round 1 not found"

    loaded_group_ids = set(round1.groups.keys())

    assert raw_group_ids == loaded_group_ids, (
        f"T2 Group IDs mismatch: raw={raw_group_ids}, loaded={loaded_group_ids}"
    )
