"""
Tests for participant label mapping in Session objects.

Verifies that participant labels (A-R, skipping I and O) are correctly mapped
and that Session.participant_labels maps participant_id to label correctly.

Author: Test Infrastructure
Date: 2026-01-16
"""

import pytest
import pandas as pd
from experiment_data import Session


# =====
# Expected participant labels (A-R, skipping I and O)
# =====
EXPECTED_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                   'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R']


# =====
# T1 Session tests
# =====
def test_participant_labels_match_raw(
    loaded_t1_session: Session, t1_raw_df: pd.DataFrame
):
    """Session.participant_labels mapping matches raw CSV."""
    # Get unique participant mappings from raw CSV
    raw_mappings = t1_raw_df[['participant.id_in_session', 'participant.label']].dropna()
    raw_mappings = raw_mappings.drop_duplicates()

    # Verify each raw mapping matches Session mapping
    for _, row in raw_mappings.iterrows():
        pid = int(row['participant.id_in_session'])
        expected_label = row['participant.label']

        assert pid in loaded_t1_session.participant_labels, \
            f"Participant ID {pid} not in session mapping"
        assert loaded_t1_session.participant_labels[pid] == expected_label, \
            f"Mismatch for ID {pid}: expected '{expected_label}', " \
            f"got '{loaded_t1_session.participant_labels[pid]}'"


def test_participant_id_to_label_mapping(
    loaded_t1_session: Session, t1_raw_df: pd.DataFrame
):
    """Mapping from id_in_session to label is correct."""
    # Build expected mapping from raw data
    raw_df = t1_raw_df[['participant.id_in_session', 'participant.label']].dropna()
    expected_mapping = dict(zip(
        raw_df['participant.id_in_session'].astype(int),
        raw_df['participant.label']
    ))

    # Session mapping should match expected
    assert loaded_t1_session.participant_labels == expected_mapping


def test_player_label_consistent_across_rounds(loaded_t1_session: Session):
    """Same participant has same label throughout all rounds."""
    # Track participant_id -> label across all rounds
    observed_labels = {}

    for segment in loaded_t1_session.segments.values():
        for round_obj in segment.rounds.values():
            for label, player in round_obj.players.items():
                pid = player.participant_id

                if pid in observed_labels:
                    assert observed_labels[pid] == label, \
                        f"Participant {pid} has inconsistent labels: " \
                        f"'{observed_labels[pid]}' and '{label}'"
                else:
                    observed_labels[pid] = label


def test_sixteen_participants(loaded_t1_session: Session):
    """Exactly 16 participants in session."""
    assert len(loaded_t1_session.participant_labels) == 16


def test_unique_participant_labels(loaded_t1_session: Session):
    """All labels are unique (A-R, no I or O)."""
    labels = list(loaded_t1_session.participant_labels.values())

    # Check uniqueness
    assert len(labels) == len(set(labels)), "Duplicate labels found"

    # Check labels are expected set (A-R, no I or O)
    assert set(labels) == set(EXPECTED_LABELS), \
        f"Labels mismatch: expected {sorted(EXPECTED_LABELS)}, " \
        f"got {sorted(labels)}"


# =====
# T2 Session tests
# =====
def test_participants_t2(loaded_t2_session: Session, t2_raw_df: pd.DataFrame):
    """Same participant verification for T2 session."""
    # Verify 16 participants
    assert len(loaded_t2_session.participant_labels) == 16, \
        f"Expected 16 participants, got {len(loaded_t2_session.participant_labels)}"

    # Verify unique labels (A-R, no I or O)
    labels = list(loaded_t2_session.participant_labels.values())
    assert len(labels) == len(set(labels)), "T2: Duplicate labels found"
    assert set(labels) == set(EXPECTED_LABELS), \
        f"T2 labels mismatch: expected {sorted(EXPECTED_LABELS)}, " \
        f"got {sorted(labels)}"

    # Verify mapping matches raw CSV
    raw_mappings = t2_raw_df[['participant.id_in_session', 'participant.label']].dropna()
    raw_mappings = raw_mappings.drop_duplicates()

    for _, row in raw_mappings.iterrows():
        pid = int(row['participant.id_in_session'])
        expected_label = row['participant.label']

        assert pid in loaded_t2_session.participant_labels, \
            f"T2: Participant ID {pid} not in session mapping"
        assert loaded_t2_session.participant_labels[pid] == expected_label, \
            f"T2: Mismatch for ID {pid}: expected '{expected_label}', " \
            f"got '{loaded_t2_session.participant_labels[pid]}'"

    # Verify label consistency across rounds
    observed_labels = {}
    for segment in loaded_t2_session.segments.values():
        for round_obj in segment.rounds.values():
            for label, player in round_obj.players.items():
                pid = player.participant_id

                if pid in observed_labels:
                    assert observed_labels[pid] == label, \
                        f"T2: Participant {pid} has inconsistent labels"
                else:
                    observed_labels[pid] = label
