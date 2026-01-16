"""
Tests for dual classifier that uses both OpenAI and Gemini.

Uses mocks to avoid actual API calls.

Author: Claude Code
Date: 2026-01-16
"""

import pytest
from unittest.mock import patch
import sys
from pathlib import Path

# Add analysis directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))

from dual_classifier import (
    classify_message_dual,
    classify_batch,
    calculate_agreement_rate,
)


# =====
# Consensus Tests - Both Agree
# =====
def test_consensus_both_agree_on_one():
    """When both classifiers return 1, consensus=1 and disputed=False."""
    with patch('dual_classifier.classify_promise_openai') as mock_openai:
        with patch('dual_classifier.classify_promise_gemini') as mock_gemini:
            mock_openai.return_value = {'classification': 1, 'raw_response': '1'}
            mock_gemini.return_value = {'classification': 1, 'raw_response': '1'}

            result = classify_message_dual("I promise to contribute 25", [])

            assert result['openai'] == 1
            assert result['gemini'] == 1
            assert result['consensus'] == 1
            assert result['disputed'] is False


def test_consensus_both_agree_on_zero():
    """When both classifiers return 0, consensus=0 and disputed=False."""
    with patch('dual_classifier.classify_promise_openai') as mock_openai:
        with patch('dual_classifier.classify_promise_gemini') as mock_gemini:
            mock_openai.return_value = {'classification': 0, 'raw_response': '0'}
            mock_gemini.return_value = {'classification': 0, 'raw_response': '0'}

            result = classify_message_dual("hello everyone", [])

            assert result['openai'] == 0
            assert result['gemini'] == 0
            assert result['consensus'] == 0
            assert result['disputed'] is False


# =====
# Disputed Tests - Classifiers Disagree
# =====
def test_disputed_openai_one_gemini_zero():
    """When OpenAI=1 and Gemini=0, consensus=None and disputed=True."""
    with patch('dual_classifier.classify_promise_openai') as mock_openai:
        with patch('dual_classifier.classify_promise_gemini') as mock_gemini:
            mock_openai.return_value = {'classification': 1, 'raw_response': '1'}
            mock_gemini.return_value = {'classification': 0, 'raw_response': '0'}

            result = classify_message_dual("maybe I'll do 25", [])

            assert result['openai'] == 1
            assert result['gemini'] == 0
            assert result['consensus'] is None
            assert result['disputed'] is True


def test_disputed_openai_zero_gemini_one():
    """When OpenAI=0 and Gemini=1, consensus=None and disputed=True."""
    with patch('dual_classifier.classify_promise_openai') as mock_openai:
        with patch('dual_classifier.classify_promise_gemini') as mock_gemini:
            mock_openai.return_value = {'classification': 0, 'raw_response': '0'}
            mock_gemini.return_value = {'classification': 1, 'raw_response': '1'}

            result = classify_message_dual("we should think about 25", [])

            assert result['openai'] == 0
            assert result['gemini'] == 1
            assert result['consensus'] is None
            assert result['disputed'] is True


# =====
# Raw Response Tests
# =====
def test_raw_responses_included():
    """Result includes raw responses from both classifiers."""
    with patch('dual_classifier.classify_promise_openai') as mock_openai:
        with patch('dual_classifier.classify_promise_gemini') as mock_gemini:
            mock_openai.return_value = {
                'classification': 1,
                'raw_response': 'The answer is 1'
            }
            mock_gemini.return_value = {
                'classification': 1,
                'raw_response': '1 - promise detected'
            }

            result = classify_message_dual("I commit to 25", [])

            assert result['openai_raw'] == 'The answer is 1'
            assert result['gemini_raw'] == '1 - promise detected'


# =====
# Context Handling Tests
# =====
def test_context_passed_to_both_classifiers():
    """Context is passed to both underlying classifiers."""
    context = [{'sender': 'A', 'body': 'lets all do 25'}]

    with patch('dual_classifier.classify_promise_openai') as mock_openai:
        with patch('dual_classifier.classify_promise_gemini') as mock_gemini:
            mock_openai.return_value = {'classification': 1, 'raw_response': '1'}
            mock_gemini.return_value = {'classification': 1, 'raw_response': '1'}

            classify_message_dual("ok deal", context)

            mock_openai.assert_called_once_with("ok deal", context)
            mock_gemini.assert_called_once_with("ok deal", context)


def test_none_context_converted_to_empty_list():
    """None context is converted to empty list."""
    with patch('dual_classifier.classify_promise_openai') as mock_openai:
        with patch('dual_classifier.classify_promise_gemini') as mock_gemini:
            mock_openai.return_value = {'classification': 0, 'raw_response': '0'}
            mock_gemini.return_value = {'classification': 0, 'raw_response': '0'}

            classify_message_dual("hello", None)

            mock_openai.assert_called_once_with("hello", [])
            mock_gemini.assert_called_once_with("hello", [])


# =====
# Agreement Rate Tests
# =====
def test_agreement_rate_all_agree():
    """Agreement rate is 1.0 when all results agree."""
    results = [
        {'disputed': False, 'consensus': 1},
        {'disputed': False, 'consensus': 0},
        {'disputed': False, 'consensus': 1},
    ]
    assert calculate_agreement_rate(results) == 1.0


def test_agreement_rate_none_agree():
    """Agreement rate is 0.0 when no results agree."""
    results = [
        {'disputed': True, 'consensus': None},
        {'disputed': True, 'consensus': None},
    ]
    assert calculate_agreement_rate(results) == 0.0


def test_agreement_rate_partial():
    """Agreement rate correctly calculates partial agreement."""
    results = [
        {'disputed': False, 'consensus': 1},
        {'disputed': True, 'consensus': None},
        {'disputed': False, 'consensus': 0},
        {'disputed': True, 'consensus': None},
    ]
    # 2 out of 4 agree = 0.5
    assert calculate_agreement_rate(results) == 0.5


def test_agreement_rate_empty_list():
    """Agreement rate returns 0.0 for empty results."""
    assert calculate_agreement_rate([]) == 0.0


# =====
# Batch Processing Tests
# =====
def test_batch_processes_all_messages():
    """Batch processing handles all messages."""
    messages = [
        {'message': 'hello', 'context': []},
        {'message': 'lets do 25', 'context': []},
        {'message': 'ok', 'context': [{'sender': 'A', 'body': 'lets do 25'}]},
    ]

    with patch('dual_classifier.classify_promise_openai') as mock_openai:
        with patch('dual_classifier.classify_promise_gemini') as mock_gemini:
            mock_openai.return_value = {'classification': 0, 'raw_response': '0'}
            mock_gemini.return_value = {'classification': 0, 'raw_response': '0'}

            results = classify_batch(messages)

            assert len(results) == 3
            assert mock_openai.call_count == 3
            assert mock_gemini.call_count == 3


def test_batch_returns_correct_structure():
    """Batch results have correct dict structure."""
    messages = [{'message': 'test', 'context': []}]

    with patch('dual_classifier.classify_promise_openai') as mock_openai:
        with patch('dual_classifier.classify_promise_gemini') as mock_gemini:
            mock_openai.return_value = {'classification': 1, 'raw_response': '1'}
            mock_gemini.return_value = {'classification': 1, 'raw_response': '1'}

            results = classify_batch(messages)

            assert 'openai' in results[0]
            assert 'gemini' in results[0]
            assert 'consensus' in results[0]
            assert 'disputed' in results[0]
            assert 'openai_raw' in results[0]
            assert 'gemini_raw' in results[0]


def test_batch_logs_progress(capsys):
    """Batch processing logs progress every 10 messages."""
    messages = [{'message': f'msg{i}', 'context': []} for i in range(25)]

    with patch('dual_classifier.classify_promise_openai') as mock_openai:
        with patch('dual_classifier.classify_promise_gemini') as mock_gemini:
            mock_openai.return_value = {'classification': 0, 'raw_response': '0'}
            mock_gemini.return_value = {'classification': 0, 'raw_response': '0'}

            classify_batch(messages)

            captured = capsys.readouterr()
            assert "10/25" in captured.out
            assert "20/25" in captured.out
            assert "25/25" in captured.out


def test_batch_logs_agreement_rate(capsys):
    """Batch processing logs final agreement rate."""
    messages = [
        {'message': 'msg1', 'context': []},
        {'message': 'msg2', 'context': []},
    ]

    with patch('dual_classifier.classify_promise_openai') as mock_openai:
        with patch('dual_classifier.classify_promise_gemini') as mock_gemini:
            mock_openai.return_value = {'classification': 0, 'raw_response': '0'}
            mock_gemini.return_value = {'classification': 0, 'raw_response': '0'}

            classify_batch(messages)

            captured = capsys.readouterr()
            assert "Agreement rate:" in captured.out
            assert "100.0%" in captured.out


def test_batch_handles_empty_list():
    """Batch processing handles empty message list."""
    results = classify_batch([])
    assert results == []


def test_batch_handles_missing_keys():
    """Batch processing handles items with missing keys."""
    messages = [
        {'message': 'test'},  # Missing 'context'
        {},  # Missing both
    ]

    with patch('dual_classifier.classify_promise_openai') as mock_openai:
        with patch('dual_classifier.classify_promise_gemini') as mock_gemini:
            mock_openai.return_value = {'classification': 0, 'raw_response': '0'}
            mock_gemini.return_value = {'classification': 0, 'raw_response': '0'}

            results = classify_batch(messages)

            assert len(results) == 2
