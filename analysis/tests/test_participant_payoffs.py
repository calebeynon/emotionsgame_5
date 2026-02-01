"""
Tests for participant_payoffs.py derived data script.

Verifies that the participant payoff extraction correctly:
- Calculates supergame payoffs as sum of round payoffs
- Extracts treatment from filenames
- Excludes participants with NaN final_payoff
- Produces deterministic output

Author: Validation Agent
Date: 2026-01-30
"""

import pandas as pd
import pytest
from pathlib import Path


# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / 'datastore' / 'derived'
RAW_DIR = Path(__file__).parent.parent / 'datastore' / 'raw'
OUTPUT_FILE = DERIVED_DIR / 'participant_payoffs.csv'

# SUPERGAME ROUND COUNTS (must match script)
SUPERGAME_ROUNDS = {1: 3, 2: 4, 3: 3, 4: 7, 5: 5}


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def output_df():
    """Load the participant payoffs output file."""
    if not OUTPUT_FILE.exists():
        pytest.skip(f"Output file not found: {OUTPUT_FILE}")
    return pd.read_csv(OUTPUT_FILE)


@pytest.fixture(scope="module")
def raw_data():
    """Load all raw data files into a dict keyed by session_code."""
    if not RAW_DIR.exists():
        pytest.skip(f"Raw directory not found: {RAW_DIR}")

    data = {}
    for f in RAW_DIR.glob('*_data.csv'):
        df = pd.read_csv(f)
        for session in df['session.code'].unique():
            session_df = df[df['session.code'] == session]
            data[session] = (session_df, f.name)
    return data


# =====
# Schema validation tests
# =====
def test_output_has_required_columns(output_df):
    """Verify output has all required columns."""
    required = [
        'participant_label', 'session_code', 'treatment',
        'total_payoff', 'sg1_payoff', 'sg2_payoff',
        'sg3_payoff', 'sg4_payoff', 'sg5_payoff'
    ]
    for col in required:
        assert col in output_df.columns, f"Missing required column: {col}"


def test_no_nan_values_in_output(output_df):
    """Verify no NaN values in any column of the output."""
    for col in output_df.columns:
        nan_count = output_df[col].isna().sum()
        assert nan_count == 0, f"Column {col} has {nan_count} NaN values"


def test_no_negative_payoffs(output_df):
    """Verify all payoff columns are non-negative."""
    payoff_cols = ['total_payoff', 'sg1_payoff', 'sg2_payoff',
                   'sg3_payoff', 'sg4_payoff', 'sg5_payoff']
    for col in payoff_cols:
        neg_count = (output_df[col] < 0).sum()
        assert neg_count == 0, f"Column {col} has {neg_count} negative values"


def test_valid_participant_labels(output_df):
    """Verify participant labels are valid (A-R, skipping I and O)."""
    valid_labels = set('ABCDEFGHJKLMNPQR')
    for label in output_df['participant_label'].unique():
        assert label in valid_labels, f"Invalid participant label: {label}"


def test_treatment_values(output_df):
    """Verify treatment is either 1 or 2."""
    valid_treatments = {1, 2}
    for t in output_df['treatment'].unique():
        assert t in valid_treatments, f"Invalid treatment value: {t}"


def test_no_duplicate_participant_session(output_df):
    """Verify no duplicate participant+session combinations."""
    duplicates = output_df.duplicated(
        subset=['participant_label', 'session_code'], keep=False
    )
    assert duplicates.sum() == 0, \
        f"Found {duplicates.sum()} duplicate participant+session combinations"


# =====
# Data integrity tests
# =====
def test_supergame_payoff_calculation(output_df, raw_data):
    """Verify supergame payoffs are correctly calculated as sum of rounds."""
    # Sample verification for first 5 participants
    for _, row in output_df.head(5).iterrows():
        session = row['session_code']
        label = row['participant_label']

        if session not in raw_data:
            continue

        raw_df, _ = raw_data[session]
        raw_row = raw_df[raw_df['participant.label'] == label]
        if len(raw_row) == 0:
            continue
        raw_row = raw_row.iloc[0]

        # Verify each supergame
        for sg in range(1, 6):
            expected = 0.0
            for r in range(1, SUPERGAME_ROUNDS[sg] + 1):
                col = f'supergame{sg}.{r}.player.payoff'
                val = raw_row.get(col, 0)
                if pd.notna(val):
                    expected += val

            actual = row[f'sg{sg}_payoff']
            assert actual == pytest.approx(expected), \
                f"SG{sg} mismatch for {label}/{session}: got {actual}, expected {expected}"


def test_total_payoff_matches_final_payoff(output_df, raw_data):
    """Verify total_payoff matches finalresults.1.player.final_payoff."""
    for _, row in output_df.head(10).iterrows():
        session = row['session_code']
        label = row['participant_label']

        if session not in raw_data:
            continue

        raw_df, _ = raw_data[session]
        raw_row = raw_df[raw_df['participant.label'] == label]
        if len(raw_row) == 0:
            continue
        raw_row = raw_row.iloc[0]

        expected = raw_row['finalresults.1.player.final_payoff']
        actual = row['total_payoff']

        assert actual == pytest.approx(expected), \
            f"total_payoff mismatch for {label}/{session}: got {actual}, expected {expected}"


def test_incomplete_participants_excluded(raw_data, output_df):
    """Verify participants with NaN final_payoff are excluded."""
    output_sessions = set(output_df['session_code'].unique())

    for session, (raw_df, filename) in raw_data.items():
        # Check for NaN final_payoff in raw data
        nan_rows = raw_df[raw_df['finalresults.1.player.final_payoff'].isna()]
        if len(nan_rows) > 0:
            # These participants should NOT be in output
            for _, raw_row in nan_rows.iterrows():
                label = raw_row['participant.label']
                if pd.notna(label):
                    match = output_df[
                        (output_df['session_code'] == session) &
                        (output_df['participant_label'] == label)
                    ]
                    assert len(match) == 0, \
                        f"Incomplete participant {label}/{session} should be excluded"


# =====
# Treatment extraction tests
# =====
def test_treatment_extraction_t1(raw_data, output_df):
    """Verify treatment=1 for sessions from _t1_ files."""
    for session, (raw_df, filename) in raw_data.items():
        if '_t1_' in filename:
            session_rows = output_df[output_df['session_code'] == session]
            if len(session_rows) > 0:
                assert (session_rows['treatment'] == 1).all(), \
                    f"Session {session} from {filename} should have treatment=1"


def test_treatment_extraction_t2(raw_data, output_df):
    """Verify treatment=2 for sessions from _t2_ files."""
    for session, (raw_df, filename) in raw_data.items():
        if '_t2_' in filename:
            session_rows = output_df[output_df['session_code'] == session]
            if len(session_rows) > 0:
                assert (session_rows['treatment'] == 2).all(), \
                    f"Session {session} from {filename} should have treatment=2"
