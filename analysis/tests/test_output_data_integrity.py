"""
Test that promise classification output data matches source experiment data.

Verifies that contribution, payoff, and message data flow correctly from
experiment_data.py through the classification pipeline to the output CSV.

Author: Claude Code
Date: 2026-01-17
"""

import sys
import json
from pathlib import Path

import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_data import load_experiment_data

# FILE PATHS
DATA_DIR = Path(__file__).parent.parent / 'datastore'
RAW_DIR = DATA_DIR / 'raw'
OUTPUT_FILE = DATA_DIR / 'derived' / 'promise_classifications.csv'


# =====
# Main test function
# =====
def test_output_data_integrity():
    """Verify output CSV matches source experiment data."""
    csv_df = load_output_csv()
    experiment = load_source_data()

    verify_all_rows(csv_df, experiment)


# =====
# Data loading
# =====
def load_output_csv() -> pd.DataFrame:
    """Load the promise classifications output CSV."""
    assert OUTPUT_FILE.exists(), f"Output file not found: {OUTPUT_FILE}"
    return pd.read_csv(OUTPUT_FILE)


def load_source_data():
    """Load source experiment data."""
    file_pairs = build_file_pairs()
    return load_experiment_data(file_pairs, name="Test Verification")


def build_file_pairs() -> list:
    """Build list of (data_csv, chat_csv, treatment) tuples."""
    file_pairs = []
    for data_file in sorted(RAW_DIR.glob("*_data.csv")):
        treatment = 1 if '_t1_' in data_file.name else 2
        chat_file = data_file.with_name(data_file.name.replace("_data", "_chat"))
        chat_path = str(chat_file) if chat_file.exists() else None
        file_pairs.append((str(data_file), chat_path, treatment))
    return file_pairs


# =====
# Verification
# =====
def verify_all_rows(csv_df: pd.DataFrame, experiment) -> None:
    """Verify all rows in CSV match source data."""
    mismatches = []

    for idx, row in csv_df.iterrows():
        player = get_source_player(experiment, row)

        if player is None:
            mismatches.append(
                f"Row {idx}: Could not find player {row['label']} "
                f"in {row['session_code']}/{row['segment']}/round {row['round']}"
            )
            continue

        # Get source messages
        group = get_player_group(experiment, row)
        source_messages = get_player_messages(group, row['label'])
        csv_messages = json.loads(row['messages'])

        # Check each field
        check_field(mismatches, idx, 'contribution', row['contribution'], player.contribution)
        check_field(mismatches, idx, 'payoff', row['payoff'], player.payoff)
        check_field(mismatches, idx, 'messages', csv_messages, source_messages)

    assert len(mismatches) == 0, (
        f"Found {len(mismatches)} data integrity issues:\n" +
        "\n".join(mismatches[:10])
    )


def get_source_player(experiment, row):
    """Get player object from source data."""
    session = experiment.get_session(row['session_code'])
    segment = session.segments[row['segment']]
    round_obj = segment.rounds[row['round']]

    for group in round_obj.groups.values():
        if row['label'] in group.players:
            return group.players[row['label']]
    return None


def get_player_group(experiment, row):
    """Get group object containing player."""
    session = experiment.get_session(row['session_code'])
    segment = session.segments[row['segment']]
    round_obj = segment.rounds[row['round']]

    for group in round_obj.groups.values():
        if row['label'] in group.players:
            return group
    return None


def get_player_messages(group, label: str) -> list:
    """Get all messages sent by player in group."""
    all_msgs = sorted(group.chat_messages, key=lambda m: m.timestamp)
    return [m.body for m in all_msgs if m.nickname == label]


def check_field(mismatches: list, row_idx: int, field_name: str, csv_value, source_value):
    """Check if CSV field matches source value."""
    if csv_value != source_value:
        mismatches.append(
            f"Row {row_idx}: {field_name} mismatch - "
            f"CSV={csv_value}, Source={source_value}"
        )


# =====
# Summary statistics test
# =====
def test_output_summary_statistics():
    """Verify output has expected summary statistics."""
    csv_df = load_output_csv()

    # Check we have data
    assert len(csv_df) > 0, "Output CSV is empty"

    # Check expected number of sessions (10 sessions in raw data)
    assert csv_df['session_code'].nunique() == 10, (
        f"Expected 10 sessions, found {csv_df['session_code'].nunique()}"
    )

    # Check we have all expected columns
    expected_columns = [
        'session_code', 'treatment', 'segment', 'round', 'group', 'label',
        'participant_id', 'contribution', 'payoff', 'message_count',
        'messages', 'classifications', 'promise_count', 'promise_percentage'
    ]
    for col in expected_columns:
        assert col in csv_df.columns, f"Missing expected column: {col}"

    # Check data types
    assert csv_df['contribution'].dtype in ['float64', 'int64'], "Contribution should be numeric"
    assert csv_df['payoff'].dtype in ['float64', 'int64'], "Payoff should be numeric"
    assert csv_df['message_count'].dtype in ['int64'], "Message count should be integer"
    assert csv_df['promise_count'].dtype in ['int64'], "Promise count should be integer"

    # Check no negative values
    assert (csv_df['contribution'] >= 0).all(), "Found negative contributions"
    assert (csv_df['message_count'] >= 0).all(), "Found negative message counts"
    assert (csv_df['promise_count'] >= 0).all(), "Found negative promise counts"

    # Check promise count <= message count
    assert (csv_df['promise_count'] <= csv_df['message_count']).all(), (
        "Promise count exceeds message count"
    )


if __name__ == "__main__":
    # Run tests
    test_output_data_integrity()
    test_output_summary_statistics()
    print("✓ All data integrity tests passed!")
