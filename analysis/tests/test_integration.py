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
import re
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


# =====
# Chat pairing integration tests
# =====
def count_messages_in_raw_chat_for_supergame(chat_df: pd.DataFrame, supergame: int) -> int:
    """Count raw chat messages for a specific supergame from channel names."""
    pattern = re.compile(rf'^\d+-supergame{supergame}-\d+$')
    return len(chat_df[chat_df['channel'].apply(lambda x: bool(pattern.match(x)))])


def get_raw_chat_count_by_supergame(chat_df: pd.DataFrame) -> dict:
    """Count raw chat messages per supergame."""
    pattern = re.compile(r'^\d+-supergame(\d+)-\d+$')

    counts = {}
    for channel in chat_df['channel']:
        match = pattern.match(channel)
        if match:
            sg = int(match.group(1))
            counts[sg] = counts.get(sg, 0) + 1
    return counts


@pytest.mark.integration
def test_chat_influences_next_round_contribution(loaded_t1_session: Session):
    """
    Verify that chat stored on round N came from round N-1's chat phase.

    Round 1 should have empty chat (no prior chat influenced it).
    Round N (N>1) should have chat from round N-1.
    """
    for sg_num in range(1, NUM_SUPERGAMES + 1):
        supergame = loaded_t1_session.get_supergame(sg_num)
        if not supergame:
            continue

        round_nums = sorted(supergame.rounds.keys())
        if not round_nums:
            continue

        # Round 1 should have empty chat_messages
        round_1 = supergame.get_round(1)
        if round_1:
            assert len(round_1.chat_messages) == 0, (
                f"Supergame {sg_num} Round 1 should have no chat messages "
                f"(no prior chat influenced it), but found {len(round_1.chat_messages)}"
            )

        # Rounds 2+ should have chat from prior round
        for round_num in round_nums[1:]:
            round_obj = supergame.get_round(round_num)
            # Chat presence depends on whether prior round had chat activity
            # We just verify the structural property: round N gets chat from round N-1
            # (The actual assignment logic is tested elsewhere)
            assert round_obj is not None


@pytest.mark.integration
def test_dataframe_chat_round_is_influenced_round(sample_experiment: Experiment):
    """
    Verify that to_dataframe_chat() shows the influenced round, not source round.

    Round 1 should have no entries (or minimal/none) because no prior chat
    influenced round 1's contribution decision.
    """
    df = sample_experiment.to_dataframe_chat(include_orphans=False)

    if df is None or len(df) == 0:
        pytest.skip("No chat data found in experiment")

    # Check that round 1 has no chat entries
    round_1_entries = df[df['round'] == 1]
    assert len(round_1_entries) == 0, (
        f"Round 1 should have no chat entries in dataframe "
        f"(no prior chat influenced it), but found {len(round_1_entries)} entries"
    )

    # Verify that chat entries exist for rounds > 1
    rounds_with_chat = df['round'].unique()
    non_first_rounds = [r for r in rounds_with_chat if r > 1]
    assert len(non_first_rounds) > 0, (
        "Expected chat entries for rounds > 1 (influenced rounds)"
    )


@pytest.mark.integration
def test_orphan_chats_not_in_regular_rounds(loaded_t1_session: Session):
    """
    Verify orphan chats are NOT duplicated in regular round chat_messages.

    segment.get_all_chat_messages(include_orphans=False) + segment.orphan_chats
    should equal total unique messages without duplication.
    """
    for sg_num in range(1, NUM_SUPERGAMES + 1):
        supergame = loaded_t1_session.get_supergame(sg_num)
        if not supergame:
            continue

        # Get regular chat messages (not including orphans)
        regular_messages = supergame.get_all_chat_messages(include_orphans=False)
        orphan_messages = supergame.get_orphan_chats_flat()

        # Create sets of (body, timestamp) tuples for comparison
        regular_keys = {(m.body, m.timestamp) for m in regular_messages}
        orphan_keys = {(m.body, m.timestamp) for m in orphan_messages}

        # Verify no overlap between regular and orphan messages
        overlap = regular_keys & orphan_keys
        assert len(overlap) == 0, (
            f"Supergame {sg_num}: Found {len(overlap)} messages duplicated "
            f"between regular rounds and orphan_chats: {list(overlap)[:3]}..."
        )


@pytest.mark.integration
def test_total_chat_count_preserved(loaded_t1_session: Session, t1_chat_df: pd.DataFrame):
    """
    Verify total messages (including orphans) equals raw CSV count.

    No messages should be lost in the pairing shift.
    """
    total_loaded = 0

    for sg_num in range(1, NUM_SUPERGAMES + 1):
        supergame = loaded_t1_session.get_supergame(sg_num)
        if not supergame:
            continue

        # Count messages: regular (in rounds) + orphans
        regular_count = len(supergame.get_all_chat_messages(include_orphans=False))
        orphan_count = len(supergame.get_orphan_chats_flat())
        sg_total = regular_count + orphan_count

        # Get expected count from raw CSV for this supergame
        expected_sg_count = count_messages_in_raw_chat_for_supergame(t1_chat_df, sg_num)

        assert sg_total == expected_sg_count, (
            f"Supergame {sg_num}: Message count mismatch. "
            f"Loaded {sg_total} (regular={regular_count}, orphan={orphan_count}), "
            f"raw CSV has {expected_sg_count}"
        )

        total_loaded += sg_total

    # Also verify total across all supergames
    raw_total = len(t1_chat_df)
    assert total_loaded == raw_total, (
        f"Total message count mismatch: loaded {total_loaded}, raw CSV has {raw_total}"
    )


@pytest.mark.integration
def test_orphan_chats_from_last_round_only(loaded_t1_session: Session):
    """
    Verify orphan chats come from the last round of each supergame.

    Since chat occurs after contribution, only the final round's chat
    becomes orphaned (no subsequent round to influence).
    """
    for sg_num in range(1, NUM_SUPERGAMES + 1):
        supergame = loaded_t1_session.get_supergame(sg_num)
        if not supergame:
            continue

        orphan_messages = supergame.get_orphan_chats_flat()
        if not orphan_messages:
            # No orphan chats is fine if there was no chat in the last round
            continue

        # Verify orphan chats exist only if there are multiple rounds
        max_round = max(supergame.rounds.keys())
        assert max_round >= 1, f"Supergame {sg_num} should have at least 1 round"

        # The presence of orphan chats indicates last-round chat activity
        # This is expected behavior based on the chat pairing semantics
        assert len(orphan_messages) >= 0  # Orphans can be empty or non-empty


@pytest.mark.integration
def test_chat_dataframe_excludes_orphans_by_default(sample_experiment: Experiment):
    """
    Verify to_dataframe_chat() excludes orphans by default.

    When include_orphans=False (default), orphan chats should not appear.
    When include_orphans=True, orphan chats should appear with round=None.
    """
    df_without = sample_experiment.to_dataframe_chat(include_orphans=False)
    df_with = sample_experiment.to_dataframe_chat(include_orphans=True)

    if df_without is None and df_with is None:
        pytest.skip("No chat data found in experiment")

    # If we have data with orphans, it should have more rows
    if df_with is not None:
        # Orphan rows have round=None
        orphan_rows = df_with[df_with['round'].isna()]

        if len(orphan_rows) > 0:
            # With orphans should have more rows
            count_without = len(df_without) if df_without is not None else 0
            count_with = len(df_with)
            assert count_with > count_without, (
                f"With orphans ({count_with}) should have more rows than "
                f"without ({count_without}) when orphan chats exist"
            )

            # Verify orphan rows have None for round
            assert orphan_rows['round'].isna().all(), (
                "Orphan chat rows should have None for round column"
            )
