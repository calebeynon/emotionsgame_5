"""
Tests for compute_sentiment.py VADER sentiment analysis functions.

Tests key helper functions for sentiment computation pipeline.

Author: Claude Code
Date: 2026-01-26
"""

import pytest
import sys
from pathlib import Path

# Add derived directory to path (where compute_sentiment.py lives)
sys.path.insert(0, str(Path(__file__).parent.parent / "derived"))

from compute_sentiment import (
    compute_std,
    aggregate_sentiment_scores,
    compute_player_sentiment,
    ID_COLS,
    PRESERVE_COLS,
)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd


# =====
# Test compute_std
# =====
class TestComputeStd:
    """Tests for the compute_std function."""

    def test_returns_zero_for_single_value(self):
        """Single value should have zero std."""
        assert compute_std([0.5]) == 0.0
        assert compute_std([1.0]) == 0.0

    def test_returns_zero_for_empty_list(self):
        """Empty list should return zero std."""
        assert compute_std([]) == 0.0

    def test_computes_sample_std(self):
        """Should compute sample standard deviation (ddof=1)."""
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        result = compute_std(values)
        # Sample std of [2,4,4,4,5,5,7,9] is approximately 2.138
        assert abs(result - 2.138) < 0.01

    def test_two_values(self):
        """Should correctly compute std for two values."""
        values = [0.0, 1.0]
        result = compute_std(values)
        # Sample std of [0, 1] with ddof=1 is sqrt(0.5) ~= 0.707
        assert abs(result - 0.707) < 0.01


# =====
# Test aggregate_sentiment_scores
# =====
class TestAggregateSentimentScores:
    """Tests for the aggregate_sentiment_scores function."""

    def test_aggregates_single_score(self):
        """Should handle single score correctly."""
        scores = [{'compound': 0.5, 'pos': 0.3, 'neg': 0.1, 'neu': 0.6}]

        result = aggregate_sentiment_scores(scores)

        assert result['sentiment_compound_mean'] == 0.5
        assert result['sentiment_compound_std'] == 0.0
        assert result['sentiment_compound_min'] == 0.5
        assert result['sentiment_compound_max'] == 0.5
        assert result['sentiment_positive_mean'] == 0.3
        assert result['sentiment_negative_mean'] == 0.1
        assert result['sentiment_neutral_mean'] == 0.6

    def test_aggregates_multiple_scores(self):
        """Should correctly aggregate multiple scores."""
        scores = [
            {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0},
            {'compound': 0.5, 'pos': 0.5, 'neg': 0.0, 'neu': 0.5},
            {'compound': -0.5, 'pos': 0.0, 'neg': 0.5, 'neu': 0.5},
        ]

        result = aggregate_sentiment_scores(scores)

        # Mean of [0.0, 0.5, -0.5] = 0.0
        assert result['sentiment_compound_mean'] == 0.0
        assert result['sentiment_compound_min'] == -0.5
        assert result['sentiment_compound_max'] == 0.5
        # Std of [0, 0.5, -0.5] with ddof=1 = 0.5
        assert abs(result['sentiment_compound_std'] - 0.5) < 0.01
        # Mean of positive [0, 0.5, 0] = 0.167
        assert abs(result['sentiment_positive_mean'] - 0.167) < 0.01

    def test_handles_extreme_values(self):
        """Should handle extreme compound values."""
        scores = [
            {'compound': 1.0, 'pos': 1.0, 'neg': 0.0, 'neu': 0.0},
            {'compound': -1.0, 'pos': 0.0, 'neg': 1.0, 'neu': 0.0},
        ]

        result = aggregate_sentiment_scores(scores)

        assert result['sentiment_compound_mean'] == 0.0
        assert result['sentiment_compound_min'] == -1.0
        assert result['sentiment_compound_max'] == 1.0


# =====
# Test compute_player_sentiment
# =====
class TestComputePlayerSentiment:
    """Tests for the compute_player_sentiment function."""

    @pytest.fixture
    def sia(self):
        """Return VADER sentiment analyzer instance."""
        return SentimentIntensityAnalyzer()

    @pytest.fixture
    def sample_row(self):
        """Return sample row matching input CSV structure."""
        return pd.Series({
            'session_code': 'test123',
            'treatment': 1,
            'segment': 'supergame1',
            'round': 2,
            'group': 1,
            'label': 'A',
            'participant_id': 1,
            'contribution': 25.0,
            'payoff': 40.0,
            'message_count': 2,
        })

    def test_preserves_identifiers(self, sia, sample_row):
        """Should preserve all identifier columns."""
        messages = ["Hello", "World"]

        result = compute_player_sentiment(sample_row, messages, sia)

        assert result['session_code'] == 'test123'
        assert result['treatment'] == 1
        assert result['segment'] == 'supergame1'
        assert result['round'] == 2
        assert result['group'] == 1
        assert result['label'] == 'A'
        assert result['participant_id'] == 1

    def test_preserves_contribution_columns(self, sia, sample_row):
        """Should preserve contribution and payoff columns."""
        messages = ["Test message"]

        result = compute_player_sentiment(sample_row, messages, sia)

        assert result['contribution'] == 25.0
        assert result['payoff'] == 40.0
        assert result['message_count'] == 2

    def test_computes_sentiment_for_positive_message(self, sia, sample_row):
        """Should compute positive sentiment for positive message."""
        messages = ["I love this! Great job everyone!"]

        result = compute_player_sentiment(sample_row, messages, sia)

        # Positive message should have positive compound
        assert result['sentiment_compound_mean'] > 0
        assert result['sentiment_positive_mean'] > 0

    def test_computes_sentiment_for_negative_message(self, sia, sample_row):
        """Should compute negative sentiment for negative message."""
        messages = ["This is terrible and awful."]

        result = compute_player_sentiment(sample_row, messages, sia)

        # Negative message should have negative compound
        assert result['sentiment_compound_mean'] < 0
        assert result['sentiment_negative_mean'] > 0

    def test_computes_sentiment_for_neutral_message(self, sia, sample_row):
        """Should compute neutral sentiment for neutral message."""
        messages = ["The experiment has 25 tokens."]

        result = compute_player_sentiment(sample_row, messages, sia)

        # Neutral message should have near-zero compound
        assert abs(result['sentiment_compound_mean']) < 0.5
        assert result['sentiment_neutral_mean'] > 0


# =====
# Test sentiment value ranges
# =====
class TestSentimentValueRanges:
    """Tests to verify sentiment values are within valid VADER ranges."""

    @pytest.fixture
    def sia(self):
        """Return VADER sentiment analyzer instance."""
        return SentimentIntensityAnalyzer()

    def test_compound_in_valid_range(self, sia):
        """Compound score should be in range [-1, 1]."""
        test_messages = [
            "I love everything about this!",
            "I hate this so much.",
            "The sky is blue.",
        ]

        for msg in test_messages:
            scores = sia.polarity_scores(msg)
            assert -1.0 <= scores['compound'] <= 1.0

    def test_components_in_valid_range(self, sia):
        """Positive, negative, neutral should be in range [0, 1]."""
        test_messages = [
            "Great wonderful amazing fantastic!",
            "Terrible horrible awful disgusting.",
            "The number is 25.",
        ]

        for msg in test_messages:
            scores = sia.polarity_scores(msg)
            assert 0.0 <= scores['pos'] <= 1.0
            assert 0.0 <= scores['neg'] <= 1.0
            assert 0.0 <= scores['neu'] <= 1.0

    def test_components_sum_to_one(self, sia):
        """Positive + negative + neutral should sum to 1.0."""
        test_messages = [
            "This is a test message.",
            "I am so happy today!",
            "This makes me angry.",
        ]

        for msg in test_messages:
            scores = sia.polarity_scores(msg)
            total = scores['pos'] + scores['neg'] + scores['neu']
            assert abs(total - 1.0) < 0.01


# =====
# Test column definitions
# =====
class TestColumnDefinitions:
    """Tests to verify column definitions are correct."""

    def test_id_cols_complete(self):
        """ID columns should include all identifiers."""
        expected = [
            'session_code', 'treatment', 'segment', 'round',
            'group', 'label', 'participant_id'
        ]
        assert ID_COLS == expected

    def test_preserve_cols_complete(self):
        """Preserve columns should include contribution data."""
        expected = ['contribution', 'payoff', 'message_count']
        assert PRESERVE_COLS == expected
