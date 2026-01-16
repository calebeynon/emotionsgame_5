"""
Tests for chat message loading and data integrity.

Verifies that ChatMessage objects loaded by experiment_data.py
correctly match the raw CSV data.

Author: Test Infrastructure
Date: 2026-01-15
"""

import pytest
import pandas as pd
from typing import List

from experiment_data import Session, ChatMessage


# =====
# Helper functions
# =====
def get_all_chat_messages(session: Session) -> List[ChatMessage]:
    """Extract all chat messages from a session."""
    all_messages = []
    for segment in session.segments.values():
        if segment.name.startswith('supergame'):
            all_messages.extend(segment.get_all_chat_messages())
    return all_messages


def find_message_in_session(session: Session, body: str) -> ChatMessage:
    """Find a specific message by its body text."""
    for msg in get_all_chat_messages(session):
        if msg.body == body:
            return msg
    return None


# =====
# T1 Session Tests
# =====
def test_chat_message_body_matches_raw(loaded_t1_session: Session, t1_chat_df: pd.DataFrame):
    """Verify ChatMessage.body matches raw CSV body column."""
    all_messages = get_all_chat_messages(loaded_t1_session)

    # Get unique bodies from raw CSV
    raw_bodies = set(t1_chat_df['body'].tolist())
    loaded_bodies = set(msg.body for msg in all_messages)

    # Check that all loaded bodies exist in raw data
    for body in loaded_bodies:
        assert body in raw_bodies, f"Loaded body '{body}' not found in raw CSV"

    # Spot check a few specific messages
    sample_raw_bodies = t1_chat_df['body'].head(5).tolist()
    for raw_body in sample_raw_bodies:
        found = find_message_in_session(loaded_t1_session, raw_body)
        assert found is not None, f"Raw body '{raw_body}' not loaded"
        assert found.body == raw_body


def test_chat_message_timestamp_matches_raw(loaded_t1_session: Session, t1_chat_df: pd.DataFrame):
    """Verify ChatMessage.timestamp matches raw CSV timestamp."""
    all_messages = get_all_chat_messages(loaded_t1_session)

    # Create lookup of body -> timestamp from raw data
    raw_timestamps = {}
    for _, row in t1_chat_df.iterrows():
        body = row['body']
        timestamp = float(row['timestamp'])
        if body not in raw_timestamps:
            raw_timestamps[body] = []
        raw_timestamps[body].append(timestamp)

    # Verify each loaded message has matching timestamp
    for msg in all_messages:
        assert msg.body in raw_timestamps, f"Body '{msg.body}' not in raw data"
        valid_timestamps = raw_timestamps[msg.body]
        assert msg.timestamp in valid_timestamps, (
            f"Timestamp {msg.timestamp} not found for body '{msg.body}'. "
            f"Expected one of: {valid_timestamps}"
        )


def test_chat_message_nickname_matches_raw(loaded_t1_session: Session, t1_chat_df: pd.DataFrame):
    """Verify ChatMessage.nickname matches raw CSV nickname."""
    all_messages = get_all_chat_messages(loaded_t1_session)

    # Create lookup of (body, timestamp) -> nickname from raw data
    raw_nicknames = {}
    for _, row in t1_chat_df.iterrows():
        key = (row['body'], float(row['timestamp']))
        raw_nicknames[key] = row['nickname']

    # Verify each loaded message has matching nickname
    for msg in all_messages:
        key = (msg.body, msg.timestamp)
        assert key in raw_nicknames, f"Message not found in raw data: {key}"
        expected_nickname = raw_nicknames[key]
        assert msg.nickname == expected_nickname, (
            f"Nickname mismatch for '{msg.body}': "
            f"got '{msg.nickname}', expected '{expected_nickname}'"
        )


def test_chat_message_count_matches_raw(loaded_t1_session: Session, t1_chat_df: pd.DataFrame):
    """Total messages loaded equals CSV row count."""
    all_messages = get_all_chat_messages(loaded_t1_session)
    loaded_count = len(all_messages)
    raw_count = len(t1_chat_df)

    assert loaded_count == raw_count, (
        f"Message count mismatch: loaded {loaded_count}, raw CSV has {raw_count}"
    )


# =====
# T2 Session Tests
# =====

# Known session code mismatches that have been verified to be the same session
# via timestamp alignment and participant label matching.
# Format: (data_session_code, chat_session_code)
VERIFIED_SESSION_CODE_EXCEPTIONS = {
    ('irrzlgk2', 'z8dowljr'),  # 03_t2: verified same session via timeline analysis
}


def test_chat_sample_t2(loaded_t2_session: Session, t2_chat_df: pd.DataFrame):
    """Same verifications for t2 session.

    Note: This test may be skipped if chat CSV and data CSV have different
    session codes (data mismatch in test files), unless the mismatch is
    in VERIFIED_SESSION_CODE_EXCEPTIONS.
    """
    all_messages = get_all_chat_messages(loaded_t2_session)

    # Check if chat CSV matches the loaded session
    chat_session_code = t2_chat_df['session_code'].iloc[0]
    loaded_session_code = loaded_t2_session.session_code

    if chat_session_code != loaded_session_code:
        # Check if this is a known verified exception
        if (loaded_session_code, chat_session_code) not in VERIFIED_SESSION_CODE_EXCEPTIONS:
            pytest.skip(
                f"Session code mismatch: chat CSV has '{chat_session_code}', "
                f"data CSV has '{loaded_session_code}'. "
                "Chat and data files may be from different sessions."
            )

    # Verify message count matches
    loaded_count = len(all_messages)
    raw_count = len(t2_chat_df)
    assert loaded_count == raw_count, (
        f"T2 message count mismatch: loaded {loaded_count}, raw CSV has {raw_count}"
    )

    # Create lookups from raw data
    raw_data = {}
    for _, row in t2_chat_df.iterrows():
        key = (row['body'], float(row['timestamp']))
        raw_data[key] = {
            'nickname': row['nickname'],
            'body': row['body'],
            'timestamp': float(row['timestamp'])
        }

    # Verify body, timestamp, and nickname for each message
    for msg in all_messages:
        key = (msg.body, msg.timestamp)
        assert key in raw_data, f"T2 message not found in raw data: {key}"

        expected = raw_data[key]
        assert msg.body == expected['body'], (
            f"T2 body mismatch: got '{msg.body}', expected '{expected['body']}'"
        )
        assert msg.timestamp == expected['timestamp'], (
            f"T2 timestamp mismatch for '{msg.body}'"
        )
        assert msg.nickname == expected['nickname'], (
            f"T2 nickname mismatch for '{msg.body}': "
            f"got '{msg.nickname}', expected '{expected['nickname']}'"
        )
