"""
Tests for verify_promises.py — interactive promise verification tool.

Author: Claude Code
Date: 2026-04-10
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "derived"))

from verify_promises import (
    build_context_lookup,
    build_item,
    build_review_items,
    compute_row_verification,
    export_results,
    find_target_index,
    make_key,
    print_summary,
    record_review,
)


# =====
# Helpers
# =====
def make_csv_row(**overrides):
    """Build a dict mimicking a CSV row for testing."""
    defaults = {
        "session_code": "test_session",
        "treatment": 1,
        "segment": "supergame1",
        "round": 2,
        "group": 1,
        "label": "A",
        "participant_id": 1,
        "contribution": 25.0,
        "payoff": 40.0,
        "message_count": 2,
        "promise_count": 1,
        "promise_percentage": 50.0,
        "messages": json.dumps(["I will give 25", "Ok"]),
        "classifications": json.dumps([1, 0]),
    }
    defaults.update(overrides)
    return defaults


def make_df(rows=None):
    """Build a DataFrame from row dicts."""
    if rows is None:
        rows = [make_csv_row()]
    return pd.DataFrame(rows)


def make_chat_msg(nickname, body, timestamp=0):
    """Build a mock ChatMessage."""
    return MagicMock(nickname=nickname, body=body, timestamp=timestamp)


def _mock_experiment(sess_code, seg_name, round_num, group_id, msgs):
    """Build a minimal experiment mock with one group."""
    group = MagicMock(chat_messages=msgs)
    round_obj = MagicMock(groups={group_id: group})
    segment = MagicMock(rounds={round_num: round_obj})
    session = MagicMock(segments={seg_name: segment})
    return MagicMock(sessions={sess_code: session})


# =====
# Test make_key and build_item
# =====
class TestKeyAndItem:
    """Tests for key generation and item construction."""

    def test_produces_pipe_separated_key(self):
        """Key should contain all identifying fields separated by pipes."""
        assert make_key(make_csv_row(), 0) == "test_session|supergame1|2|1|A|0"

    def test_different_msg_idx_produces_different_key(self):
        """Keys should differ for different message indices."""
        row = make_csv_row()
        assert make_key(row, 0) != make_key(row, 1)

    def test_different_labels_produce_different_keys(self):
        """Keys should differ for different player labels."""
        assert make_key(make_csv_row(label="A"), 0) != make_key(make_csv_row(label="B"), 0)

    def test_item_captures_all_fields(self):
        """Item should contain key, indices, message, and metadata."""
        item = build_item(make_csv_row(), row_idx=5, msg_idx=1, msg="Ok", cls=0)
        assert item["key"] == "test_session|supergame1|2|1|A|1"
        assert item["row_idx"] == 5
        assert item["msg_idx"] == 1
        assert item["message"] == "Ok"
        assert item["original_classification"] == 0

    def test_round_and_group_are_int(self):
        """Round and group should be integers for context lookup compatibility."""
        item = build_item(make_csv_row(round=3, group=2), 0, 0, "msg", 1)
        assert isinstance(item["round"], int)
        assert isinstance(item["group"], int)


# =====
# Test build_review_items
# =====
class TestBuildReviewItems:
    """Tests for flattening CSV into review items."""

    def test_single_row_two_messages(self):
        """Row with 2 messages should produce 2 review items."""
        df = make_df()
        items = build_review_items(df)
        assert len(items) == 2

    def test_preserves_message_order(self):
        """Items should follow message order within each row."""
        df = make_df()
        items = build_review_items(df)
        assert items[0]["message"] == "I will give 25"
        assert items[1]["message"] == "Ok"

    def test_multiple_rows(self):
        """Multiple rows should all be flattened."""
        rows = [
            make_csv_row(label="A", messages=json.dumps(["msg1"])),
            make_csv_row(label="B", messages=json.dumps(["msg2", "msg3"])),
        ]
        df = make_df(rows)
        items = build_review_items(df)
        assert len(items) == 3

    def test_empty_messages_row(self):
        """Row with empty messages array should produce 0 items."""
        rows = [make_csv_row(messages=json.dumps([]), classifications=json.dumps([]))]
        df = make_df(rows)
        items = build_review_items(df)
        assert len(items) == 0

    def test_all_keys_unique(self):
        """All review item keys must be unique."""
        rows = [
            make_csv_row(label="A", messages=json.dumps(["m1", "m2"])),
            make_csv_row(label="B", messages=json.dumps(["m3"])),
        ]
        df = make_df(rows)
        items = build_review_items(df)
        keys = [i["key"] for i in items]
        assert len(keys) == len(set(keys))


# =====
# Test find_target_index
# =====
class TestFindTargetIndex:
    """Tests for finding a player's N-th message in group conversation."""

    def test_finds_first_message(self):
        """Should return index 0 for player's first message at start."""
        msgs = [make_chat_msg("A", "Hello", 1), make_chat_msg("B", "Hi", 2)]
        assert find_target_index(msgs, "A", 0) == 0

    def test_finds_second_message_of_player(self):
        """Should return correct global index for player's second message."""
        msgs = [
            make_chat_msg("A", "Hello", 1),
            make_chat_msg("B", "Hi", 2),
            make_chat_msg("A", "How are you?", 3),
        ]
        assert find_target_index(msgs, "A", 1) == 2

    def test_returns_none_for_out_of_range(self):
        """Should return None when msg_idx exceeds player's message count."""
        msgs = [make_chat_msg("A", "Hello", 1)]
        assert find_target_index(msgs, "A", 5) is None

    def test_returns_none_for_missing_player(self):
        """Should return None when player not in conversation."""
        msgs = [make_chat_msg("A", "Hello", 1)]
        assert find_target_index(msgs, "C", 0) is None

    def test_returns_none_for_empty_list(self):
        """Should return None for empty message list."""
        assert find_target_index([], "A", 0) is None

    def test_multiple_players_interleaved(self):
        """Should correctly count per-player indices in interleaved messages."""
        msgs = [
            make_chat_msg("A", "a1", 1),
            make_chat_msg("B", "b1", 2),
            make_chat_msg("C", "c1", 3),
            make_chat_msg("A", "a2", 4),
            make_chat_msg("B", "b2", 5),
        ]
        assert find_target_index(msgs, "B", 0) == 1
        assert find_target_index(msgs, "B", 1) == 4
        assert find_target_index(msgs, "C", 0) == 2


# =====
# Test record_review
# =====
class TestRecordReview:
    """Tests for recording review decisions."""

    def test_records_confirmed_decision(self):
        """Confirming should store original classification unchanged."""
        progress = {}
        item = {"key": "k1", "original_classification": 1}
        record_review(progress, item, verified_cls=1)
        assert progress["k1"]["verified"] == 1
        assert progress["k1"]["original"] == 1
        assert progress["k1"]["changed"] is False

    def test_records_flipped_decision(self):
        """Flipping should store new classification and mark changed."""
        progress = {}
        item = {"key": "k1", "original_classification": 1}
        record_review(progress, item, verified_cls=0)
        assert progress["k1"]["verified"] == 0
        assert progress["k1"]["original"] == 1
        assert progress["k1"]["changed"] is True


# =====
# Test compute_row_verification
# =====
class TestComputeRowVerification:
    """Tests for computing verified classifications per CSV row."""

    def test_unreviewed_row(self):
        """Row with no reviews should return 'unreviewed' status."""
        row = make_csv_row()
        cls, status, changes = compute_row_verification(row, {})
        assert status == "unreviewed"
        assert changes == 0
        assert json.loads(cls) == [1, 0]  # unchanged originals

    def test_partially_reviewed_row(self):
        """Row with some reviews should return 'partial' status."""
        row = make_csv_row()
        key_0 = make_key(row, 0)
        progress = {key_0: {"verified": 0, "original": 1, "changed": True}}
        cls, status, changes = compute_row_verification(row, progress)
        assert status == "partial"
        assert changes == 1
        assert json.loads(cls) == [0, 0]  # first flipped, second unchanged

    def test_fully_reviewed_row(self):
        """Row with all reviews should return 'verified' status."""
        row = make_csv_row()
        progress = {
            make_key(row, 0): {"verified": 1, "original": 1, "changed": False},
            make_key(row, 1): {"verified": 1, "original": 0, "changed": True},
        }
        cls, status, changes = compute_row_verification(row, progress)
        assert status == "verified"
        assert changes == 1
        assert json.loads(cls) == [1, 1]

    def test_single_message_row(self):
        """Row with one message should work correctly."""
        row = make_csv_row(
            messages=json.dumps(["single"]),
            classifications=json.dumps([0]),
        )
        key = make_key(row, 0)
        progress = {key: {"verified": 0, "original": 0, "changed": False}}
        cls, status, changes = compute_row_verification(row, progress)
        assert status == "verified"
        assert changes == 0


# =====
# Test export_results
# =====
class TestExportResults:
    """Tests for CSV export logic."""

    def test_no_export_on_empty_progress(self, capsys):
        """Should print message and not create file when progress is empty."""
        df = make_df()
        export_results(df, {})
        captured = capsys.readouterr()
        assert "No reviews to export" in captured.out

    def test_export_creates_correct_columns(self, tmp_path, monkeypatch):
        """Exported CSV should have verification columns."""
        output_file = tmp_path / "verified.csv"
        monkeypatch.setattr(
            "verify_promises.OUTPUT_FILE", output_file
        )
        df = make_df()
        row = df.iloc[0]
        key_0 = make_key(row, 0)
        progress = {key_0: {"verified": 0, "original": 1, "changed": True}}

        export_results(df, progress)

        result = pd.read_csv(output_file)
        assert "verified_classifications" in result.columns
        assert "review_status" in result.columns
        assert "num_changes" in result.columns
        assert result.iloc[0]["review_status"] == "partial"
        assert result.iloc[0]["num_changes"] == 1


# =====
# Test build_context_lookup
# =====
class TestBuildContextLookup:
    """Tests for experiment data context lookup construction."""

    def test_builds_lookup_from_experiment(self):
        """Should create keyed lookup from experiment data."""
        experiment = _mock_experiment("sess1", "supergame1", 2, 1, [make_chat_msg("A", "Hello", 1.0)])
        lookup = build_context_lookup(experiment)
        key = ("sess1", "supergame1", 2, 1)
        assert key in lookup
        assert len(lookup[key]) == 1
        assert lookup[key][0].body == "Hello"

    def test_skips_non_supergame_segments(self):
        """Should not include introduction or finalresults segments."""
        seg = MagicMock(rounds={})
        session = MagicMock(segments={"introduction": seg, "finalresults": seg})
        experiment = MagicMock(sessions={"sess1": session})
        assert len(build_context_lookup(experiment)) == 0

    def test_sorts_messages_by_timestamp(self):
        """Messages in lookup should be sorted by timestamp."""
        msgs = [make_chat_msg("B", "Second", 2.0), make_chat_msg("A", "First", 1.0)]
        experiment = _mock_experiment("s1", "supergame2", 1, 1, msgs)
        result = build_context_lookup(experiment)[("s1", "supergame2", 1, 1)]
        assert result[0].timestamp < result[1].timestamp


# =====
# Test print_summary
# =====
class TestPrintSummary:
    """Tests for summary statistics output."""

    def test_summary_with_all_confirmed(self, capsys):
        """Should report 100% agreement when nothing changed."""
        progress = {
            "k1": {"verified": 1, "original": 1, "changed": False},
            "k2": {"verified": 0, "original": 0, "changed": False},
        }
        print_summary(progress)
        out = capsys.readouterr().out
        assert "Confirmed (agree):  2" in out
        assert "Changed (disagree): 0" in out

    def test_summary_with_flips(self, capsys):
        """Should report correct flip counts by direction."""
        progress = {
            "k1": {"verified": 1, "original": 0, "changed": True},  # 0->1
            "k2": {"verified": 0, "original": 1, "changed": True},  # 1->0
            "k3": {"verified": 1, "original": 1, "changed": False},
        }
        print_summary(progress)
        out = capsys.readouterr().out
        assert "Changed (disagree): 2" in out
        assert "0→1 (missed promises): 1" in out  # noqa: RUF001
        assert "1→0 (false promises):  1" in out  # noqa: RUF001

    def test_summary_handles_empty_progress(self, capsys):
        """Should not crash on empty progress (division by zero guard)."""
        print_summary({})
        out = capsys.readouterr().out
        assert "Messages reviewed:  0" in out
