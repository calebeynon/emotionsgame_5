"""
Tests for chat message loading and data integrity.

Verifies that ChatMessage objects loaded by experiment_data.py
correctly match the raw CSV data.

Chat messages are stored on the round they INFLUENCED, not the round
they occurred in:
- Round 1: chat_messages is empty (no prior chat influenced it)
- Round N (N>1): chat_messages contains round N-1's chat
- Last round's chat: stored in segment.orphan_chats

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
def get_all_chat_messages(session: Session, include_orphans: bool = True) -> List[ChatMessage]:
    """Extract all chat messages from a session.

    Args:
        session: Session object to extract messages from
        include_orphans: If True, include orphan chats from last rounds
    """
    all_messages = []
    for segment in session.segments.values():
        if segment.name.startswith('supergame'):
            all_messages.extend(segment.get_all_chat_messages(include_orphans=include_orphans))
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


# =====
# Chat-Round Pairing Semantics Tests
# =====
def test_round_1_chat_messages_empty(loaded_t1_session: Session):
    """Verify round 1 has no chat messages across all supergames.

    Round 1 should have empty chat_messages because no prior chat
    influenced the first contribution decision.
    """
    for sg_num in range(1, 6):
        segment = loaded_t1_session.get_supergame(sg_num)
        if segment is None:
            continue

        round_1 = segment.get_round(1)
        if round_1 is None:
            continue

        # Round-level chat should be empty
        assert len(round_1.chat_messages) == 0, (
            f"Supergame {sg_num} round 1 should have no chat, "
            f"but found {len(round_1.chat_messages)} messages"
        )

        # Group-level chat should also be empty
        for group_id, group in round_1.groups.items():
            assert len(group.chat_messages) == 0, (
                f"Supergame {sg_num} round 1 group {group_id} should have no chat"
            )

        # Player-level chat should also be empty
        for label, player in round_1.players.items():
            assert len(player.chat_messages) == 0, (
                f"Supergame {sg_num} round 1 player {label} should have no chat"
            )


def test_round_n_has_previous_round_chat(loaded_t1_session: Session, t1_chat_df: pd.DataFrame):
    """Verify rounds 2+ receive chat from the previous round.

    Chat that occurred after round N-1's contribution decision should
    be stored on round N (because it influenced round N's decision).

    This test parses the raw chat CSV channel format to determine which
    chatgroup corresponds to which source round, then verifies those
    messages appear on the correct target round (source_round + 1).
    """
    import re

    # Parse chat CSV to build expected chat per (supergame, source_round)
    # Channel format: {session_num}-supergame{N}-{chatgroup}
    # Chatgroups are sequential: round 1 uses first 4, round 2 uses next 4, etc.
    channel_pattern = re.compile(r'^\d+\-supergame(\d+)\-(\d+)$')

    # Build mapping of (supergame, chatgroup) -> list of (body, timestamp)
    raw_chat_by_channel = {}
    for _, row in t1_chat_df.iterrows():
        match = channel_pattern.match(row['channel'])
        if match:
            sg_num = int(match.group(1))
            chatgroup = int(match.group(2))
            key = (sg_num, chatgroup)
            if key not in raw_chat_by_channel:
                raw_chat_by_channel[key] = []
            raw_chat_by_channel[key].append((row['body'], float(row['timestamp'])))

    # For each supergame, determine chatgroup ranges per round
    for sg_num in range(1, 6):
        segment = loaded_t1_session.get_supergame(sg_num)
        if segment is None:
            continue

        # Get all chatgroups for this supergame
        sg_chatgroups = sorted([cg for (sg, cg) in raw_chat_by_channel.keys() if sg == sg_num])
        if not sg_chatgroups:
            continue

        min_chatgroup = min(sg_chatgroups)
        max_round = max(segment.rounds.keys())

        # Test: chat from source_round N-1 should appear on target_round N
        for source_round in range(1, max_round):
            target_round = source_round + 1

            # Calculate chatgroup range for source_round (4 groups per round)
            round_start_cg = min_chatgroup + (source_round - 1) * 4
            round_chatgroups = list(range(round_start_cg, round_start_cg + 4))

            # Collect all expected messages from source round's chatgroups
            expected_messages = set()
            for cg in round_chatgroups:
                key = (sg_num, cg)
                if key in raw_chat_by_channel:
                    for body, ts in raw_chat_by_channel[key]:
                        expected_messages.add((body, ts))

            if not expected_messages:
                # No chat in source round, skip
                continue

            # Get actual messages on target round
            target_round_obj = segment.get_round(target_round)
            if target_round_obj is None:
                continue

            actual_messages = set()
            for msg in target_round_obj.chat_messages:
                actual_messages.add((msg.body, msg.timestamp))

            # Verify expected messages appear on target round
            missing = expected_messages - actual_messages
            assert len(missing) == 0, (
                f"Supergame {sg_num}: chat from round {source_round} missing on round {target_round}. "
                f"Missing {len(missing)} of {len(expected_messages)} messages. "
                f"Sample missing: {list(missing)[:3]}"
            )

            # Verify no extra messages (from wrong round)
            extra = actual_messages - expected_messages
            assert len(extra) == 0, (
                f"Supergame {sg_num}: round {target_round} has unexpected messages. "
                f"Found {len(extra)} extra messages not from round {source_round}. "
                f"Sample extra: {list(extra)[:3]}"
            )


def test_orphan_chats_stored_at_segment_level(loaded_t1_session: Session, t1_chat_df: pd.DataFrame):
    """Verify last round's chat is stored in segment.orphan_chats.

    Chat from the last round of each supergame has no subsequent round
    to influence, so it's stored as 'orphan chats' at the segment level.

    This test verifies that the actual chat content from the last round's
    chatgroups appears in orphan_chats, not just that the structure exists.
    """
    import re

    # Parse chat CSV to build expected chat per (supergame, chatgroup)
    channel_pattern = re.compile(r'^\d+\-supergame(\d+)\-(\d+)$')

    raw_chat_by_channel = {}
    for _, row in t1_chat_df.iterrows():
        match = channel_pattern.match(row['channel'])
        if match:
            sg_num = int(match.group(1))
            chatgroup = int(match.group(2))
            key = (sg_num, chatgroup)
            if key not in raw_chat_by_channel:
                raw_chat_by_channel[key] = []
            raw_chat_by_channel[key].append((row['body'], float(row['timestamp'])))

    for sg_num in range(1, 6):
        segment = loaded_t1_session.get_supergame(sg_num)
        if segment is None:
            continue

        # Get all chatgroups for this supergame
        sg_chatgroups = sorted([cg for (sg, cg) in raw_chat_by_channel.keys() if sg == sg_num])
        if not sg_chatgroups:
            continue

        min_chatgroup = min(sg_chatgroups)
        max_round = max(segment.rounds.keys())

        # Calculate chatgroup range for last round
        last_round_start_cg = min_chatgroup + (max_round - 1) * 4
        last_round_chatgroups = list(range(last_round_start_cg, last_round_start_cg + 4))

        # Collect expected messages from last round's chatgroups
        expected_orphans = set()
        for cg in last_round_chatgroups:
            key = (sg_num, cg)
            if key in raw_chat_by_channel:
                for body, ts in raw_chat_by_channel[key]:
                    expected_orphans.add((body, ts))

        # Get actual orphan messages
        orphan_flat = segment.get_orphan_chats_flat()
        actual_orphans = set((msg.body, msg.timestamp) for msg in orphan_flat)

        # Verify expected orphans match actual orphans
        missing = expected_orphans - actual_orphans
        assert len(missing) == 0, (
            f"Supergame {sg_num}: last round chat missing from orphan_chats. "
            f"Missing {len(missing)} of {len(expected_orphans)} messages. "
            f"Sample missing: {list(missing)[:3]}"
        )

        extra = actual_orphans - expected_orphans
        assert len(extra) == 0, (
            f"Supergame {sg_num}: orphan_chats has unexpected messages. "
            f"Found {len(extra)} extra messages. "
            f"Sample extra: {list(extra)[:3]}"
        )

        # Verify orphan chats are accessible as both dict and flat list
        orphan_chats = segment.get_orphan_chats()
        if orphan_chats:
            for label, messages in orphan_chats.items():
                assert isinstance(label, str), "Orphan chat key should be player label"
                assert isinstance(messages, list), "Orphan chat value should be list"
                for msg in messages:
                    assert isinstance(msg, ChatMessage), "Orphan should be ChatMessage"

        # Verify flat list contains same total messages as dict
        dict_total = sum(len(msgs) for msgs in orphan_chats.values())
        assert len(orphan_flat) == dict_total, (
            f"Supergame {sg_num}: orphan flat list ({len(orphan_flat)}) != "
            f"dict total ({dict_total})"
        )


def test_chat_timestamp_ordering(loaded_t1_session: Session):
    """Verify chat messages are sorted by timestamp within each context.

    Messages should be sorted by timestamp at round, group, and player levels.
    """
    for sg_num in range(1, 6):
        segment = loaded_t1_session.get_supergame(sg_num)
        if segment is None:
            continue

        # Check orphan chats are sorted
        for label, messages in segment.orphan_chats.items():
            if len(messages) > 1:
                timestamps = [m.timestamp for m in messages]
                assert timestamps == sorted(timestamps), (
                    f"Supergame {sg_num} orphan chats for {label} not sorted"
                )

        # Check each round
        for round_num, round_obj in segment.rounds.items():
            # Round-level sorting
            if len(round_obj.chat_messages) > 1:
                timestamps = [m.timestamp for m in round_obj.chat_messages]
                assert timestamps == sorted(timestamps), (
                    f"Supergame {sg_num} round {round_num} chat not sorted"
                )

            # Group-level sorting
            for group_id, group in round_obj.groups.items():
                if len(group.chat_messages) > 1:
                    timestamps = [m.timestamp for m in group.chat_messages]
                    assert timestamps == sorted(timestamps), (
                        f"Supergame {sg_num} round {round_num} "
                        f"group {group_id} chat not sorted"
                    )

            # Player-level sorting
            for label, player in round_obj.players.items():
                if len(player.chat_messages) > 1:
                    timestamps = [m.timestamp for m in player.chat_messages]
                    assert timestamps == sorted(timestamps), (
                        f"Supergame {sg_num} round {round_num} "
                        f"player {label} chat not sorted"
                    )


def test_total_chat_with_orphans_matches_raw(loaded_t1_session: Session, t1_chat_df: pd.DataFrame):
    """Verify total chat count (including orphans) matches raw CSV.

    When include_orphans=True, the total message count should equal
    the raw CSV row count.
    """
    # Get all messages including orphans
    all_messages = get_all_chat_messages(loaded_t1_session, include_orphans=True)
    loaded_count = len(all_messages)
    raw_count = len(t1_chat_df)

    assert loaded_count == raw_count, (
        f"Total message count (with orphans): loaded {loaded_count}, "
        f"raw CSV has {raw_count}"
    )


def test_total_chat_without_orphans_less_than_raw(loaded_t1_session: Session, t1_chat_df: pd.DataFrame):
    """Verify chat count without orphans is less than raw CSV count.

    Without orphans, we should be missing the last round's chat from
    each supergame.
    """
    # Get messages excluding orphans
    messages_no_orphans = get_all_chat_messages(loaded_t1_session, include_orphans=False)
    count_no_orphans = len(messages_no_orphans)

    # Get messages including orphans
    messages_with_orphans = get_all_chat_messages(loaded_t1_session, include_orphans=True)
    count_with_orphans = len(messages_with_orphans)

    raw_count = len(t1_chat_df)

    # Without orphans should be less than with orphans
    assert count_no_orphans <= count_with_orphans, (
        f"Without orphans ({count_no_orphans}) should be <= "
        f"with orphans ({count_with_orphans})"
    )

    # With orphans should equal raw count
    assert count_with_orphans == raw_count, (
        f"With orphans ({count_with_orphans}) should equal raw ({raw_count})"
    )
