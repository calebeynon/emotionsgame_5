"""
Integration tests for full data loading pipeline.

End-to-end tests that load real session data and verify data integrity
across the complete experiment data structure.

Author: Test Infrastructure
Date: 2026-01-16
"""

import pytest
import pandas as pd
import random
from experiment_data import Session, Experiment


# =====
# Constants
# =====
EXPECTED_CONTRIBUTION_COLUMNS = [
    'session_code', 'treatment', 'segment', 'round', 'group',
    'label', 'participant_id', 'contribution', 'payoff', 'role'
]
NUM_SUPERGAMES = 5


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


def count_valid_contribution_rows(raw_df: pd.DataFrame) -> int:
    """
    Count the number of valid contribution data points in raw CSV.

    Each participant has contributions across 5 supergames, each with multiple rounds.
    """
    total = 0
    valid_participants = raw_df[raw_df['participant.label'].notna()]

    for sg in range(1, NUM_SUPERGAMES + 1):
        round_num = 1
        while True:
            col = f"supergame{sg}.{round_num}.player.contribution"
            if col not in raw_df.columns:
                break
            # Count non-null contributions for valid participants
            count = valid_participants[col].notna().sum()
            total += count
            round_num += 1

    return total


# =====
# Full session load tests
# =====
@pytest.mark.integration
def test_full_session_load_t1(t1_session_paths: tuple):
    """Load complete t1 session without errors."""
    from experiment_data import load_experiment_data

    data_path, chat_path = t1_session_paths
    chat_str = str(chat_path) if chat_path.exists() else None

    file_pairs = [(str(data_path), chat_str, 1)]
    experiment = load_experiment_data(file_pairs, name="T1 Load Test")

    # Verify experiment loaded
    assert experiment is not None
    assert len(experiment.sessions) == 1

    # Verify session structure
    session_codes = experiment.list_session_codes()
    assert len(session_codes) == 1

    session = experiment.get_session(session_codes[0])
    assert session is not None
    assert session.treatment == 1
    assert len(session.segments) > 0

    # Verify supergames exist
    for sg_num in range(1, NUM_SUPERGAMES + 1):
        supergame = session.get_supergame(sg_num)
        assert supergame is not None, f"Supergame {sg_num} not found"
        assert len(supergame.rounds) > 0, f"Supergame {sg_num} has no rounds"


@pytest.mark.integration
def test_full_session_load_t2(t2_session_paths: tuple):
    """Load complete t2 session without errors."""
    from experiment_data import load_experiment_data

    data_path, chat_path = t2_session_paths
    chat_str = str(chat_path) if chat_path.exists() else None

    file_pairs = [(str(data_path), chat_str, 2)]
    experiment = load_experiment_data(file_pairs, name="T2 Load Test")

    # Verify experiment loaded
    assert experiment is not None
    assert len(experiment.sessions) == 1

    # Verify session structure
    session_codes = experiment.list_session_codes()
    assert len(session_codes) == 1

    session = experiment.get_session(session_codes[0])
    assert session is not None
    assert session.treatment == 2
    assert len(session.segments) > 0

    # Verify supergames exist
    for sg_num in range(1, NUM_SUPERGAMES + 1):
        supergame = session.get_supergame(sg_num)
        assert supergame is not None, f"Supergame {sg_num} not found"
        assert len(supergame.rounds) > 0, f"Supergame {sg_num} has no rounds"


# =====
# Experiment-level tests
# =====
@pytest.mark.integration
def test_experiment_with_both_sessions(sample_experiment: Experiment):
    """Load both sessions into Experiment object."""
    # Verify experiment contains both sessions
    assert sample_experiment is not None
    assert len(sample_experiment.sessions) == 2

    session_codes = sample_experiment.list_session_codes()
    assert len(session_codes) == 2

    # Verify both treatments are present
    treatments = set()
    for code in session_codes:
        session = sample_experiment.get_session(code)
        treatments.add(session.treatment)

    assert 1 in treatments, "Treatment 1 session not found"
    assert 2 in treatments, "Treatment 2 session not found"


@pytest.mark.integration
def test_to_dataframe_contributions_has_expected_columns(sample_experiment: Experiment):
    """DataFrame has expected columns."""
    df = sample_experiment.to_dataframe_contributions()

    assert df is not None, "to_dataframe_contributions() returned None"
    assert isinstance(df, pd.DataFrame)

    # Check all expected columns exist
    missing_cols = set(EXPECTED_CONTRIBUTION_COLUMNS) - set(df.columns)
    extra_cols = set(df.columns) - set(EXPECTED_CONTRIBUTION_COLUMNS)

    assert len(missing_cols) == 0, f"Missing columns: {missing_cols}"
    assert len(extra_cols) == 0, f"Unexpected columns: {extra_cols}"


@pytest.mark.integration
def test_to_dataframe_contributions_row_count(sample_experiment: Experiment,
                                               t1_raw_df: pd.DataFrame,
                                               t2_raw_df: pd.DataFrame):
    """Row count reasonable compared to raw data."""
    df = sample_experiment.to_dataframe_contributions()
    assert df is not None

    # Count expected rows from raw data
    t1_expected = count_valid_contribution_rows(t1_raw_df)
    t2_expected = count_valid_contribution_rows(t2_raw_df)
    total_expected = t1_expected + t2_expected

    # Allow some tolerance for potential edge cases
    actual_count = len(df)

    assert actual_count > 0, "DataFrame has no rows"
    assert actual_count == total_expected, (
        f"Row count mismatch: expected {total_expected}, got {actual_count}"
    )


@pytest.mark.integration
def test_random_sample_verification(sample_experiment: Experiment,
                                    t1_raw_df: pd.DataFrame):
    """Randomly sample several data points and verify against raw data."""
    df = sample_experiment.to_dataframe_contributions()
    assert df is not None

    # Get t1 session code
    t1_session = None
    for code, session in sample_experiment.sessions.items():
        if session.treatment == 1:
            t1_session = session
            break

    assert t1_session is not None, "T1 session not found"

    # Get valid participants from raw data
    valid_labels = t1_raw_df[t1_raw_df['participant.label'].notna()]['participant.label'].unique()
    assert len(valid_labels) > 0, "No valid participants in raw data"

    # Sample random data points to verify
    random.seed(42)  # Reproducible randomness
    num_samples = min(10, len(valid_labels))
    sampled_labels = random.sample(list(valid_labels), num_samples)

    mismatches = []
    for label in sampled_labels:
        # Pick random supergame and round
        sg_num = random.randint(1, NUM_SUPERGAMES)
        round_num = random.randint(1, 3)  # Most supergames have at least 3 rounds

        raw_value = get_raw_contribution(t1_raw_df, label, sg_num, round_num)
        if raw_value is None:
            continue

        # Find corresponding value in DataFrame
        df_row = df[
            (df['session_code'] == t1_session.session_code) &
            (df['segment'] == f'supergame{sg_num}') &
            (df['round'] == round_num) &
            (df['label'] == label)
        ]

        if df_row.empty:
            mismatches.append({
                'label': label,
                'supergame': sg_num,
                'round': round_num,
                'raw': raw_value,
                'loaded': 'NOT FOUND'
            })
        else:
            loaded_value = df_row['contribution'].iloc[0]
            if raw_value != loaded_value:
                mismatches.append({
                    'label': label,
                    'supergame': sg_num,
                    'round': round_num,
                    'raw': raw_value,
                    'loaded': loaded_value
                })

    assert len(mismatches) == 0, (
        f"Found {len(mismatches)} data mismatches in random sample:\n"
        + "\n".join(
            f"  {m['label']} SG{m['supergame']} R{m['round']}: "
            f"raw={m['raw']} loaded={m['loaded']}"
            for m in mismatches
        )
    )
