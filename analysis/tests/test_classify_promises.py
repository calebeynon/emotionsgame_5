"""
Tests for classify_promises.py promise classification script.

Uses mocks for LLM calls to avoid actual API costs during testing.

Author: Claude Code
Date: 2026-01-16
"""

import json
import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))

from classify_promises import (
    build_file_pairs,
    extract_treatment,
    collect_all_messages,
    build_context,
    build_result_record,
    classify_player_round,
    calculate_avg_context,
)
import pandas as pd


# =====
# Treatment extraction tests
# =====
def test_extract_treatment_t1():
    """Extract treatment 1 from t1 filename."""
    assert extract_treatment("01_t1_data.csv") == 1


def test_extract_treatment_t2():
    """Extract treatment 2 from t2 filename."""
    assert extract_treatment("03_t2_data.csv") == 2


def test_extract_treatment_unknown():
    """Return 0 for unknown treatment pattern."""
    assert extract_treatment("unknown_data.csv") == 0


def test_build_file_pairs_returns_list():
    """build_file_pairs returns a list."""
    result = build_file_pairs()
    assert isinstance(result, list)


# =====
# Context building tests
# =====
def test_build_context_empty_for_first_message():
    """First message has empty context."""
    msg1 = MagicMock()
    msg1.body = "hello"
    msg1.nickname = "A"
    context = build_context([msg1], "hello", "A")
    assert context == []


def test_build_context_includes_prior_messages():
    """Context includes messages before target."""
    msg1, msg2 = MagicMock(), MagicMock()
    msg1.body, msg1.nickname = "first", "B"
    msg2.body, msg2.nickname = "second", "A"
    context = build_context([msg1, msg2], "second", "A")
    assert len(context) == 1
    assert context[0] == {'sender': "B", 'body': "first"}


def test_build_context_stops_at_target():
    """Context does not include messages after target."""
    msgs = []
    for body, nick in [("first", "A"), ("target", "B"), ("after", "C")]:
        m = MagicMock()
        m.body, m.nickname = body, nick
        msgs.append(m)
    context = build_context(msgs, "target", "B")
    assert len(context) == 1
    assert context[0]['body'] == "first"


def test_build_context_with_multiple_prior_messages():
    """Context includes all prior messages in order."""
    msgs = []
    for body, nick in [("one", "A"), ("two", "B"), ("three", "C"), ("target", "D")]:
        m = MagicMock()
        m.body, m.nickname = body, nick
        msgs.append(m)
    context = build_context(msgs, "target", "D")
    assert len(context) == 3
    assert [c['body'] for c in context] == ["one", "two", "three"]


# =====
# Result record building tests
# =====
def test_build_result_record_basic():
    """Build result record with correct fields."""
    player_data = {
        'session_code': 'abc123', 'treatment': 1, 'segment': 'supergame1',
        'round': 1, 'group': 1, 'label': 'A', 'participant_id': 1,
        'contribution': 25.0, 'payoff': 10.0, 'messages': ['hello', 'lets do 25'],
    }
    result = build_result_record(player_data, [0, 1], [0, 1], [0, 1], 0)

    assert result['session_code'] == 'abc123'
    assert result['message_count'] == 2
    assert result['promise_count'] == 1
    assert result['promise_percentage'] == 50.0
    assert result['disputed_count'] == 0


def test_build_result_record_json_fields():
    """Result record has valid JSON for list fields."""
    player_data = {
        'session_code': 'test', 'treatment': 1, 'segment': 'supergame1',
        'round': 1, 'group': 1, 'label': 'A', 'participant_id': 1,
        'contribution': 0, 'payoff': 0, 'messages': ['msg1', 'msg2'],
    }
    result = build_result_record(player_data, [1, 0], [1, 0], [1, 0], 0)

    assert json.loads(result['messages']) == ['msg1', 'msg2']
    assert json.loads(result['classifications']) == [1, 0]


def test_build_result_record_no_promises():
    """Result record handles zero promises correctly."""
    player_data = {
        'session_code': 'test', 'treatment': 1, 'segment': 'supergame1',
        'round': 1, 'group': 1, 'label': 'A', 'participant_id': 1,
        'contribution': 0, 'payoff': 0, 'messages': ['hello', 'hi'],
    }
    result = build_result_record(player_data, [0, 0], [0, 0], [0, 0], 0)
    assert result['promise_count'] == 0
    assert result['promise_percentage'] == 0.0


def test_build_result_record_all_promises():
    """Result record handles all promises correctly."""
    player_data = {
        'session_code': 'test', 'treatment': 1, 'segment': 'supergame1',
        'round': 1, 'group': 1, 'label': 'A', 'participant_id': 1,
        'contribution': 25, 'payoff': 10, 'messages': ['I promise 25', 'yes 25'],
    }
    result = build_result_record(player_data, [1, 1], [1, 1], [1, 1], 0)
    assert result['promise_count'] == 2
    assert result['promise_percentage'] == 100.0


def test_build_result_record_with_disputes():
    """Result record tracks disputed count."""
    player_data = {
        'session_code': 'test', 'treatment': 1, 'segment': 'supergame1',
        'round': 1, 'group': 1, 'label': 'A', 'participant_id': 1,
        'contribution': 0, 'payoff': 0, 'messages': ['maybe', 'unclear'],
    }
    result = build_result_record(player_data, [None, None], [0, 1], [1, 0], 2)
    assert result['disputed_count'] == 2


# =====
# DataFrame building tests
# =====
def test_dataframe_has_required_columns():
    """DataFrame has all required columns."""
    results = [{
        'session_code': 'test', 'treatment': 1, 'segment': 'supergame1',
        'round': 1, 'group': 1, 'label': 'A', 'participant_id': 1,
        'contribution': 25, 'payoff': 10, 'message_count': 2,
        'promise_count': 1, 'promise_percentage': 50.0, 'disputed_count': 0,
        'messages': '["hello", "25"]', 'classifications': '[0, 1]',
        'openai_classifications': '[0, 1]', 'gemini_classifications': '[0, 1]',
    }]
    df = pd.DataFrame.from_records(results)
    expected_cols = [
        'session_code', 'treatment', 'segment', 'round', 'group', 'label',
        'participant_id', 'contribution', 'payoff', 'message_count',
        'promise_count', 'promise_percentage', 'disputed_count', 'messages',
        'classifications', 'openai_classifications', 'gemini_classifications'
    ]
    for col in expected_cols:
        assert col in df.columns


def test_dataframe_multiple_rows():
    """Multiple results produce multiple DataFrame rows."""
    results = [
        {'session_code': f'test{i}', 'treatment': 1, 'segment': 'sg1',
         'round': 1, 'group': 1, 'label': chr(65 + i), 'participant_id': i,
         'contribution': 0, 'payoff': 0, 'message_count': 1, 'promise_count': 0,
         'promise_percentage': 0, 'disputed_count': 0, 'messages': '[]',
         'classifications': '[]', 'openai_classifications': '[]',
         'gemini_classifications': '[]'}
        for i in range(3)
    ]
    df = pd.DataFrame.from_records(results)
    assert len(df) == 3


# =====
# Classification with mocks tests
# =====
def test_classify_player_round_with_mock():
    """classify_player_round works with mocked dual classifier."""
    msg = MagicMock()
    msg.body, msg.nickname, msg.timestamp = "I promise 25", "A", 1000

    player_data = {
        'session_code': 'test', 'treatment': 1, 'segment': 'supergame1',
        'round': 1, 'group': 1, 'label': 'A', 'participant_id': 1,
        'contribution': 25, 'payoff': 10, 'messages': ['I promise 25'],
        'all_group_msgs': [msg],
    }
    mock_result = {'openai': 1, 'gemini': 1, 'consensus': 1, 'disputed': False}

    with patch('classify_promises.classify_message_dual', return_value=mock_result):
        result = classify_player_round(player_data)

    assert result['message_count'] == 1
    assert result['promise_count'] == 1
    assert result['disputed_count'] == 0


def test_classify_player_round_handles_disputes():
    """classify_player_round counts disputes correctly."""
    msg1, msg2 = MagicMock(), MagicMock()
    msg1.body, msg1.nickname, msg1.timestamp = "maybe", "A", 1000
    msg2.body, msg2.nickname, msg2.timestamp = "unclear", "A", 2000

    player_data = {
        'session_code': 'test', 'treatment': 1, 'segment': 'supergame1',
        'round': 1, 'group': 1, 'label': 'A', 'participant_id': 1,
        'contribution': 0, 'payoff': 0, 'messages': ['maybe', 'unclear'],
        'all_group_msgs': [msg1, msg2],
    }
    mock_result = {'openai': 0, 'gemini': 1, 'consensus': None, 'disputed': True}

    with patch('classify_promises.classify_message_dual', return_value=mock_result):
        result = classify_player_round(player_data)

    assert result['disputed_count'] == 2


# =====
# Average context calculation tests
# =====
def test_calculate_avg_context_empty():
    """Empty messages data returns 0."""
    assert calculate_avg_context([]) == 0


def test_calculate_avg_context_single_message():
    """Single first message has zero context."""
    msg = MagicMock()
    msg.nickname = "A"
    player_data = {'label': 'A', 'all_group_msgs': [msg]}
    assert calculate_avg_context([player_data]) == 0


def test_calculate_avg_context_with_prior_messages():
    """Messages with prior context calculate correctly."""
    msg1, msg2 = MagicMock(), MagicMock()
    msg1.nickname, msg2.nickname = "B", "A"
    player_data = {'label': 'A', 'all_group_msgs': [msg1, msg2]}
    assert calculate_avg_context([player_data]) == 1.0


# =====
# Message collection tests (with fixture)
# =====
def test_collect_all_messages_returns_list(sample_experiment):
    """collect_all_messages returns a list."""
    result = collect_all_messages(sample_experiment)
    assert isinstance(result, list)


def test_collect_all_messages_has_required_keys(sample_experiment):
    """Collected messages have required dictionary keys."""
    result = collect_all_messages(sample_experiment)
    if len(result) > 0:
        required_keys = [
            'session_code', 'treatment', 'segment', 'round', 'group',
            'label', 'participant_id', 'contribution', 'payoff',
            'messages', 'all_group_msgs'
        ]
        for key in required_keys:
            assert key in result[0], f"Missing key: {key}"


def test_collect_all_messages_only_supergames(sample_experiment):
    """Collected messages only come from supergame segments."""
    result = collect_all_messages(sample_experiment)
    for item in result:
        assert item['segment'].startswith('supergame')
