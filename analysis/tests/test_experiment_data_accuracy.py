"""
Validate experiment_data.py loads data correctly against raw CSV files.

Catches data loading bugs by comparing the hierarchical Experiment object
against the original oTree CSV exports in the datastore.

Author: Claude Code
Date: 2026-02-16
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'derived'))

from classify_behavior import build_file_pairs
from experiment_data import load_experiment_data

# FILE PATHS
RAW_DIR = Path(__file__).parent.parent / 'datastore' / 'raw'

# CONSTANTS
EXPECTED_LABELS = {
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
    'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
}
MULTIPLIER = 0.4
ROUNDS_PER_SUPERGAME = {1: 3, 2: 4, 3: 3, 4: 7, 5: 5}
PLAYERS_PER_GROUP = 4


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def experiment():
    """Load the full experiment once for all tests."""
    file_pairs = build_file_pairs()
    return load_experiment_data(file_pairs, name="Accuracy Test")


@pytest.fixture(scope="module")
def raw_data_frames():
    """Load all raw data CSVs keyed by session code."""
    frames = {}
    for data_file in sorted(RAW_DIR.glob("*_data.csv")):
        df = pd.read_csv(data_file)
        session_code = df['session.code'].iloc[0]
        frames[session_code] = df
    return frames


@pytest.fixture(scope="module")
def raw_chat_frames():
    """Load all raw chat CSVs keyed by session code."""
    frames = {}
    for chat_file in sorted(RAW_DIR.glob("*_chat.csv")):
        df = pd.read_csv(chat_file)
        session_code = df['session_code'].iloc[0]
        frames[session_code] = df
    return frames


# =====
# Test: contribution accuracy
# =====
def test_contribution_accuracy(experiment, raw_data_frames):
    """Verify player.contribution matches raw CSV for every player-round."""
    mismatches = []
    for code, session in experiment.sessions.items():
        raw_df = raw_data_frames[code]
        mismatches.extend(_check_contributions(session, raw_df))

    assert len(mismatches) == 0, (
        f"{len(mismatches)} contribution mismatches:\n"
        + "\n".join(mismatches[:10])
    )


def _check_contributions(session, raw_df):
    """Compare contributions for one session against raw CSV."""
    mismatches = []
    for sg_num in range(1, 6):
        segment = session.get_supergame(sg_num)
        if not segment:
            continue
        for rnd_num, rnd in segment.rounds.items():
            col = f"supergame{sg_num}.{rnd_num}.player.contribution"
            for label, player in rnd.players.items():
                raw_val = _get_raw_value(raw_df, label, col)
                if raw_val != player.contribution:
                    mismatches.append(
                        f"  {session.session_code} SG{sg_num} R{rnd_num} "
                        f"{label}: raw={raw_val} loaded={player.contribution}"
                    )
    return mismatches


# =====
# Test: payoff accuracy
# =====
def test_payoff_accuracy(experiment, raw_data_frames):
    """Verify player.payoff matches raw CSV for every player-round."""
    mismatches = []
    for code, session in experiment.sessions.items():
        raw_df = raw_data_frames[code]
        mismatches.extend(_check_payoffs(session, raw_df))

    assert len(mismatches) == 0, (
        f"{len(mismatches)} payoff mismatches:\n"
        + "\n".join(mismatches[:10])
    )


def _check_payoffs(session, raw_df):
    """Compare payoffs for one session against raw CSV."""
    mismatches = []
    for sg_num in range(1, 6):
        segment = session.get_supergame(sg_num)
        if not segment:
            continue
        for rnd_num, rnd in segment.rounds.items():
            col = f"supergame{sg_num}.{rnd_num}.player.payoff"
            for label, player in rnd.players.items():
                raw_val = _get_raw_value(raw_df, label, col)
                if raw_val != player.payoff:
                    mismatches.append(
                        f"  {session.session_code} SG{sg_num} R{rnd_num} "
                        f"{label}: raw={raw_val} loaded={player.payoff}"
                    )
    return mismatches


# =====
# Test: group totals
# =====
def test_group_total_contribution(experiment):
    """Verify group.total_contribution equals sum of member contributions."""
    mismatches = []
    for code, session in experiment.sessions.items():
        mismatches.extend(_check_group_totals(code, session))

    assert len(mismatches) == 0, (
        f"{len(mismatches)} group total mismatches:\n"
        + "\n".join(mismatches[:10])
    )


def _check_group_totals(code, session):
    """Compare group totals for one session."""
    mismatches = []
    for sg_num, rnd_num, gid, group in _iter_groups(session):
        expected = sum(
            p.contribution for p in group.players.values()
            if p.contribution is not None
        )
        if group.total_contribution != expected:
            mismatches.append(
                f"  {code} SG{sg_num} R{rnd_num} G{gid}: "
                f"stored={group.total_contribution} computed={expected}"
            )
    return mismatches


# =====
# Test: individual share matches raw CSV
# =====
def test_individual_share(experiment, raw_data_frames):
    """Verify group.individual_share matches the raw CSV value.

    oTree rounds individual_share to the nearest integer, so we compare
    against the raw CSV rather than computing total_contribution * 0.4.
    """
    mismatches = []
    for code, session in experiment.sessions.items():
        raw_df = raw_data_frames[code]
        mismatches.extend(_check_individual_shares(session, raw_df))

    assert len(mismatches) == 0, (
        f"{len(mismatches)} individual share mismatches:\n"
        + "\n".join(mismatches[:10])
    )


def _check_individual_shares(session, raw_df):
    """Compare individual shares for one session against raw CSV."""
    mismatches = []
    for sg_num, rnd_num, gid, group in _iter_groups(session):
        col = f"supergame{sg_num}.{rnd_num}.group.individual_share"
        raw_val = _get_raw_group_value(raw_df, col, sg_num, rnd_num, gid)
        if raw_val is not None and group.individual_share != raw_val:
            mismatches.append(
                f"  {session.session_code} SG{sg_num} R{rnd_num} G{gid}: "
                f"loaded={group.individual_share} raw={raw_val}"
            )
    return mismatches


# =====
# Test: player labels
# =====
def test_player_labels(experiment):
    """Verify all 16 labels (A-R, skipping I and O) present per session."""
    for code, session in experiment.sessions.items():
        labels_found = set(session.participant_labels.values())
        assert labels_found == EXPECTED_LABELS, (
            f"Session {code}: expected {EXPECTED_LABELS}, "
            f"got {labels_found}. "
            f"Missing: {EXPECTED_LABELS - labels_found}, "
            f"Extra: {labels_found - EXPECTED_LABELS}"
        )


# =====
# Test: chat message counts
# =====
def test_chat_message_counts(experiment, raw_chat_frames):
    """Verify total chat messages per session match raw chat CSV row count.

    Including orphans (last-round chats) should recover the full raw count.
    """
    mismatches = _compare_chat_counts(experiment, raw_chat_frames)
    assert len(mismatches) == 0, (
        f"{len(mismatches)} chat count mismatches:\n"
        + "\n".join(mismatches)
    )


def _compare_chat_counts(experiment, raw_chat_frames):
    """Compare loaded chat totals with raw CSV row counts."""
    mismatches = []
    for code, session in experiment.sessions.items():
        if code not in raw_chat_frames:
            continue
        raw_count = len(raw_chat_frames[code])
        loaded_count = _count_session_chats(session)
        if loaded_count != raw_count:
            mismatches.append(
                f"  {code}: raw={raw_count} loaded={loaded_count}"
            )
    return mismatches


def _count_session_chats(session):
    """Count all chat messages in a session including orphans."""
    total = 0
    for seg_name, segment in session.segments.items():
        if not seg_name.startswith('supergame'):
            continue
        total += len(segment.get_all_chat_messages(include_orphans=True))
    return total


# =====
# Test: session/segment structure
# =====
def test_rounds_per_supergame(experiment):
    """Verify correct number of rounds per supergame across all sessions."""
    for code, session in experiment.sessions.items():
        for sg_num, expected_rounds in ROUNDS_PER_SUPERGAME.items():
            segment = session.get_supergame(sg_num)
            assert segment is not None, (
                f"Session {code}: supergame{sg_num} not found"
            )
            actual = len(segment.rounds)
            assert actual == expected_rounds, (
                f"Session {code} supergame{sg_num}: "
                f"expected {expected_rounds} rounds, got {actual}"
            )


# =====
# Test: group membership
# =====
def test_four_players_per_group(experiment):
    """Verify 4 players per group in every round across all sessions."""
    violations = []
    for code, session in experiment.sessions.items():
        violations.extend(_check_group_sizes(code, session))

    assert len(violations) == 0, (
        f"{len(violations)} groups with wrong player count:\n"
        + "\n".join(violations[:10])
    )


def _check_group_sizes(code, session):
    """Check group sizes for one session."""
    violations = []
    for sg_num, rnd_num, gid, group in _iter_groups(session):
        count = len(group.players)
        if count != PLAYERS_PER_GROUP:
            violations.append(
                f"  {code} SG{sg_num} R{rnd_num} G{gid}: {count} players"
            )
    return violations


# =====
# Helpers
# =====
def _iter_groups(session):
    """Yield (sg_num, rnd_num, group_id, group) for all supergame groups."""
    for sg_num in range(1, 6):
        segment = session.get_supergame(sg_num)
        if not segment:
            continue
        for rnd_num, rnd in segment.rounds.items():
            for gid, group in rnd.groups.items():
                yield sg_num, rnd_num, gid, group


def _get_raw_value(raw_df, label, col_name):
    """Extract a numeric value from raw CSV for a participant label."""
    if col_name not in raw_df.columns:
        return None
    row = raw_df[raw_df['participant.label'] == label]
    if row.empty:
        return None
    value = row[col_name].iloc[0]
    return float(value) if pd.notna(value) else None


def _get_raw_group_value(raw_df, col_name, sg_num, rnd_num, group_id):
    """Extract a group-level value from raw CSV by matching group id."""
    if col_name not in raw_df.columns:
        return None
    gid_col = f"supergame{sg_num}.{rnd_num}.group.id_in_subsession"
    if gid_col not in raw_df.columns:
        return None
    mask = raw_df[gid_col] == group_id
    rows = raw_df.loc[mask, col_name].dropna()
    if rows.empty:
        return None
    return float(rows.iloc[0])
