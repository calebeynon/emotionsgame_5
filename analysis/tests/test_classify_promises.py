"""
Tests for classify_promises.py core functions.

Tests key helper functions for message classification pipeline.

Author: Claude Code
Date: 2026-01-17
"""

import json
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add analysis directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))

from classify_promises import (
    extract_treatment,
    build_context,
    build_result_record,
    calculate_avg_context,
    flatten_messages,
    aggregate_results,
)


# =====
# Test extract_treatment
# =====
class TestExtractTreatment:
    """Tests for the extract_treatment function."""

    def test_extracts_treatment_1(self):
        """Should extract treatment 1 from filename with _t1_."""
        assert extract_treatment("01_t1_data.csv") == 1
        assert extract_treatment("session_t1_chat.csv") == 1

    def test_extracts_treatment_2(self):
        """Should extract treatment 2 from filename with _t2_."""
        assert extract_treatment("01_t2_data.csv") == 2
        assert extract_treatment("session_t2_chat.csv") == 2

    def test_returns_zero_for_unknown_treatment(self):
        """Should return 0 when treatment marker not found."""
        assert extract_treatment("data.csv") == 0
        assert extract_treatment("session_chat.csv") == 0

    def test_handles_multiple_underscores(self):
        """Should correctly extract treatment with complex filenames."""
        assert extract_treatment("2025_01_17_t1_data_v2.csv") == 1
        assert extract_treatment("experiment_final_t2_results.csv") == 2


# =====
# Test build_context
# =====
class TestBuildContext:
    """Tests for the build_context function."""

    def test_empty_context_for_first_message(self):
        """First message should have empty context."""
        msg1 = MagicMock(nickname="A", body="First message", timestamp=1)
        context = build_context([msg1], "First message", "A")

        assert context == []

    def test_includes_prior_messages(self):
        """Context should include all messages before target."""
        msg1 = MagicMock(nickname="A", body="Hello", timestamp=1)
        msg2 = MagicMock(nickname="B", body="Hi there", timestamp=2)
        msg3 = MagicMock(nickname="A", body="How are you?", timestamp=3)

        context = build_context([msg1, msg2, msg3], "How are you?", "A")

        assert len(context) == 2
        assert context[0] == {"sender": "A", "body": "Hello"}
        assert context[1] == {"sender": "B", "body": "Hi there"}

    def test_stops_at_target_message(self):
        """Context should stop before target message."""
        msg1 = MagicMock(nickname="A", body="First", timestamp=1)
        msg2 = MagicMock(nickname="B", body="Second", timestamp=2)
        msg3 = MagicMock(nickname="C", body="Third", timestamp=3)
        msg4 = MagicMock(nickname="A", body="Fourth", timestamp=4)

        context = build_context([msg1, msg2, msg3, msg4], "Second", "B")

        assert len(context) == 1
        assert context[0]["body"] == "First"

    def test_matches_both_body_and_label(self):
        """Should match on both message body and sender label."""
        msg1 = MagicMock(nickname="A", body="test", timestamp=1)
        msg2 = MagicMock(nickname="B", body="test", timestamp=2)  # Same body, different sender

        # Context for A's "test" should be empty
        context_a = build_context([msg1, msg2], "test", "A")
        assert len(context_a) == 0

        # Context for B's "test" should include A's message
        context_b = build_context([msg1, msg2], "test", "B")
        assert len(context_b) == 1
        assert context_b[0]["sender"] == "A"


# =====
# Test build_result_record
# =====
class TestBuildResultRecord:
    """Tests for the build_result_record function."""

    def test_builds_complete_record(self):
        """Should build record with all required fields."""
        player_data = {
            "session_code": "abc123",
            "treatment": 1,
            "segment": "supergame1",
            "round": 1,
            "group": 1,
            "label": "A",
            "participant_id": 123,
            "contribution": 25,
            "payoff": 40,
            "messages": ["I'll do 25", "sounds good"],
        }
        classifications = [1, 1]

        result = build_result_record(player_data, classifications)

        assert result["session_code"] == "abc123"
        assert result["treatment"] == 1
        assert result["segment"] == "supergame1"
        assert result["round"] == 1
        assert result["group"] == 1
        assert result["label"] == "A"
        assert result["participant_id"] == 123
        assert result["contribution"] == 25
        assert result["payoff"] == 40

    def test_counts_promises_correctly(self):
        """Should count number of promises (1s) in classifications."""
        player_data = {
            "session_code": "test",
            "treatment": 1,
            "segment": "supergame1",
            "round": 1,
            "group": 1,
            "label": "A",
            "participant_id": 1,
            "contribution": 20,
            "payoff": 30,
            "messages": ["yes", "no", "ok", "maybe"],
        }
        classifications = [1, 0, 1, 0]

        result = build_result_record(player_data, classifications)

        assert result["message_count"] == 4
        assert result["promise_count"] == 2
        assert result["promise_percentage"] == 50.0

    def test_handles_zero_messages(self):
        """Should handle case with no messages."""
        player_data = {
            "session_code": "test",
            "treatment": 1,
            "segment": "supergame1",
            "round": 1,
            "group": 1,
            "label": "A",
            "participant_id": 1,
            "contribution": 0,
            "payoff": 0,
            "messages": [],
        }
        classifications = []

        result = build_result_record(player_data, classifications)

        assert result["message_count"] == 0
        assert result["promise_count"] == 0
        assert result["promise_percentage"] == 0.0

    def test_serializes_messages_and_classifications(self):
        """Should serialize messages and classifications as JSON."""
        player_data = {
            "session_code": "test",
            "treatment": 1,
            "segment": "supergame1",
            "round": 1,
            "group": 1,
            "label": "A",
            "participant_id": 1,
            "contribution": 25,
            "payoff": 40,
            "messages": ["test message"],
        }
        classifications = [1]

        result = build_result_record(player_data, classifications)

        # Should be valid JSON strings
        messages = json.loads(result["messages"])
        cls = json.loads(result["classifications"])

        assert messages == ["test message"]
        assert cls == [1]


# =====
# Test calculate_avg_context
# =====
class TestCalculateAvgContext:
    """Tests for the calculate_avg_context function."""

    def test_calculates_average_correctly(self):
        """Should calculate average context length across all messages."""
        msg1 = MagicMock(nickname="A", body="msg1")
        msg2 = MagicMock(nickname="B", body="msg2")
        msg3 = MagicMock(nickname="A", body="msg3")

        messages_data = [
            {
                "label": "A",
                "all_group_msgs": [msg1, msg2, msg3],
                "messages": ["msg1", "msg3"],
            }
        ]

        avg = calculate_avg_context(messages_data)

        # msg1 has 0 context, msg3 has 2 context
        # Average = (0 + 2) / 2 = 1.0
        assert avg == 1.0

    def test_handles_empty_messages(self):
        """Should return 0 for empty messages."""
        messages_data = []
        avg = calculate_avg_context(messages_data)
        assert avg == 0.0

    def test_multiple_players(self):
        """Should aggregate across multiple players."""
        msg1 = MagicMock(nickname="A", body="First")
        msg2 = MagicMock(nickname="B", body="Second")
        msg3 = MagicMock(nickname="A", body="Third")

        messages_data = [
            {
                "label": "A",
                "all_group_msgs": [msg1, msg2, msg3],
                "messages": ["First", "Third"],
            },
            {
                "label": "B",
                "all_group_msgs": [msg1, msg2, msg3],
                "messages": ["Second"],
            },
        ]

        avg = calculate_avg_context(messages_data)

        # A: msg "First" has 0 context, "Third" has 2 context
        # B: msg "Second" has 1 context
        # Average = (0 + 2 + 1) / 3 = 1.0
        assert avg == 1.0


# =====
# Test flatten_messages
# =====
class TestFlattenMessages:
    """Tests for the flatten_messages function."""

    def test_flattens_single_player(self):
        """Should flatten messages from single player."""
        msg1 = MagicMock(nickname="A", body="Hello")
        msg2 = MagicMock(nickname="A", body="World")

        messages_data = [
            {
                "label": "A",
                "all_group_msgs": [msg1, msg2],
                "messages": ["Hello", "World"],
            }
        ]

        flat_messages, index_map = flatten_messages(messages_data)

        assert len(flat_messages) == 2
        assert flat_messages[0]["message"] == "Hello"
        assert flat_messages[1]["message"] == "World"
        assert index_map == [0, 0]

    def test_index_map_tracks_players(self):
        """Index map should track which player each message belongs to."""
        msg1 = MagicMock(nickname="A", body="A1")
        msg2 = MagicMock(nickname="B", body="B1")
        msg3 = MagicMock(nickname="A", body="A2")

        messages_data = [
            {
                "label": "A",
                "all_group_msgs": [msg1, msg2, msg3],
                "messages": ["A1", "A2"],
            },
            {
                "label": "B",
                "all_group_msgs": [msg1, msg2, msg3],
                "messages": ["B1"],
            },
        ]

        flat_messages, index_map = flatten_messages(messages_data)

        assert len(flat_messages) == 3
        assert index_map == [0, 0, 1]  # First two from player 0, last from player 1

    def test_builds_context_for_each_message(self):
        """Each flattened message should have correct context."""
        msg1 = MagicMock(nickname="A", body="First")
        msg2 = MagicMock(nickname="B", body="Second")
        msg3 = MagicMock(nickname="A", body="Third")

        messages_data = [
            {
                "label": "A",
                "all_group_msgs": [msg1, msg2, msg3],
                "messages": ["First", "Third"],
            }
        ]

        flat_messages, index_map = flatten_messages(messages_data)

        # First message should have empty context
        assert flat_messages[0]["context"] == []

        # Third message should have 2 prior messages in context
        assert len(flat_messages[1]["context"]) == 2


# =====
# Test aggregate_results
# =====
class TestAggregateResults:
    """Tests for the aggregate_results function."""

    def test_aggregates_to_player_records(self):
        """Should aggregate flat classifications back to player-round records."""
        messages_data = [
            {
                "session_code": "test",
                "treatment": 1,
                "segment": "supergame1",
                "round": 1,
                "group": 1,
                "label": "A",
                "participant_id": 1,
                "contribution": 25,
                "payoff": 40,
                "messages": ["msg1", "msg2"],
            }
        ]
        classifications = [1, 0]
        index_map = [0, 0]

        results = aggregate_results(messages_data, classifications, index_map)

        assert len(results) == 1
        assert results[0]["promise_count"] == 1
        assert results[0]["message_count"] == 2

    def test_handles_multiple_players(self):
        """Should correctly map classifications to multiple players."""
        messages_data = [
            {
                "session_code": "test",
                "treatment": 1,
                "segment": "supergame1",
                "round": 1,
                "group": 1,
                "label": "A",
                "participant_id": 1,
                "contribution": 25,
                "payoff": 40,
                "messages": ["A1", "A2"],
            },
            {
                "session_code": "test",
                "treatment": 1,
                "segment": "supergame1",
                "round": 1,
                "group": 1,
                "label": "B",
                "participant_id": 2,
                "contribution": 20,
                "payoff": 35,
                "messages": ["B1"],
            },
        ]
        classifications = [1, 1, 0]
        index_map = [0, 0, 1]  # First two to player 0, last to player 1

        results = aggregate_results(messages_data, classifications, index_map)

        assert len(results) == 2
        assert results[0]["promise_count"] == 2  # Player A: both 1s
        assert results[1]["promise_count"] == 0  # Player B: one 0

    def test_preserves_player_order(self):
        """Should maintain original player order in results."""
        messages_data = [
            {
                "session_code": "test",
                "treatment": 1,
                "segment": "supergame1",
                "round": 1,
                "group": 1,
                "label": "A",
                "participant_id": 1,
                "contribution": 25,
                "payoff": 40,
                "messages": ["A1"],
            },
            {
                "session_code": "test",
                "treatment": 1,
                "segment": "supergame1",
                "round": 1,
                "group": 1,
                "label": "B",
                "participant_id": 2,
                "contribution": 20,
                "payoff": 35,
                "messages": ["B1"],
            },
        ]
        classifications = [1, 0]
        index_map = [0, 1]

        results = aggregate_results(messages_data, classifications, index_map)

        assert results[0]["label"] == "A"
        assert results[1]["label"] == "B"
