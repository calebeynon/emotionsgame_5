"""
Tests for the prompt_templates module.

Author: Claude Code
Date: 2026-01-16
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "llm_clients"))

from prompt_templates import (
    build_classification_prompt,
    estimate_prompt_tokens,
    PROMISE_EXAMPLES,
)


# =====
# Test build_classification_prompt
# =====
class TestBuildClassificationPrompt:
    """Tests for the build_classification_prompt function."""

    def test_prompt_includes_target_message(self):
        """Prompt should include the target message to classify."""
        message = "I'll contribute 25 points"
        context = []

        prompt = build_classification_prompt(message, context)

        assert message in prompt
        assert "TARGET MESSAGE TO CLASSIFY" in prompt

    def test_prompt_includes_conversation_context(self):
        """Prompt should include prior messages when context provided."""
        message = "sounds good"
        context = [
            {"sender": "Player A", "body": "Let's all put in 25"},
            {"sender": "Player B", "body": "I'm in"},
        ]

        prompt = build_classification_prompt(message, context)

        assert "Player A" in prompt
        assert "Let's all put in 25" in prompt
        assert "Player B" in prompt
        assert "I'm in" in prompt
        assert "CONVERSATION CONTEXT" in prompt

    def test_prompt_handles_empty_context(self):
        """Prompt should handle empty context gracefully."""
        message = "let's do 25"
        context = []

        prompt = build_classification_prompt(message, context)

        assert "No prior messages" in prompt

    def test_prompt_includes_examples(self):
        """Prompt should include promise examples."""
        message = "test message"
        context = []

        prompt = build_classification_prompt(message, context)

        assert "EXAMPLES OF PROMISES" in prompt
        # Check some specific examples are included
        assert "ok im in" in prompt
        assert "deal" in prompt
        assert "lets do it" in prompt

    def test_prompt_includes_edge_case_instructions(self):
        """Prompt should include edge case handling instructions."""
        message = "yes"
        context = []

        prompt = build_classification_prompt(message, context)

        assert "IMPORTANT - WHAT IS A PROMISE" in prompt
        assert "IMPORTANT - WHAT IS NOT A PROMISE" in prompt
        assert "okay" in prompt.lower() or "yes" in prompt.lower()
        assert "sounds good" in prompt.lower()
        assert "proposal" in prompt.lower()

    def test_prompt_includes_game_explanation(self):
        """Prompt should explain the public goods game context."""
        message = "test"
        context = []

        prompt = build_classification_prompt(message, context)

        assert "PUBLIC GOODS GAME" in prompt
        assert "25 points" in prompt or "25" in prompt
        assert "group" in prompt.lower()

    def test_prompt_asks_for_binary_response(self):
        """Prompt should ask for 0 or 1 response."""
        message = "test"
        context = []

        prompt = build_classification_prompt(message, context)

        assert "0" in prompt and "1" in prompt
        assert "single" in prompt.lower() or "only" in prompt.lower()

    def test_context_preserves_message_order(self):
        """Context messages should appear in the order provided."""
        message = "I agree"
        context = [
            {"sender": "A", "body": "First message"},
            {"sender": "B", "body": "Second message"},
            {"sender": "C", "body": "Third message"},
        ]

        prompt = build_classification_prompt(message, context)

        first_pos = prompt.find("First message")
        second_pos = prompt.find("Second message")
        third_pos = prompt.find("Third message")

        assert first_pos < second_pos < third_pos

    def test_context_handles_missing_sender(self):
        """Context should handle messages with missing sender key."""
        message = "okay"
        context = [{"body": "Let's contribute"}]

        prompt = build_classification_prompt(message, context)

        assert "Unknown" in prompt or "Let's contribute" in prompt


# =====
# Test estimate_prompt_tokens
# =====
class TestEstimatePromptTokens:
    """Tests for the estimate_prompt_tokens function."""

    def test_returns_positive_integer(self):
        """Token estimate should be a positive integer."""
        prompt = "This is a test prompt"

        tokens = estimate_prompt_tokens(prompt)

        assert isinstance(tokens, int)
        assert tokens > 0

    def test_longer_prompt_more_tokens(self):
        """Longer prompts should have more tokens."""
        short_prompt = "Short test"
        long_prompt = "This is a much longer prompt with many more words"

        short_tokens = estimate_prompt_tokens(short_prompt)
        long_tokens = estimate_prompt_tokens(long_prompt)

        assert long_tokens > short_tokens

    def test_reasonable_estimate_for_typical_prompt(self):
        """Token estimate should be reasonable for a typical classification prompt."""
        message = "let's all contribute 25"
        context = [{"sender": "A", "body": "What should we do?"}]

        prompt = build_classification_prompt(message, context)
        tokens = estimate_prompt_tokens(prompt)

        # A typical prompt should be between 200-1000 tokens
        assert 100 < tokens < 2000

    def test_empty_prompt_returns_zero(self):
        """Empty prompt should return zero tokens."""
        tokens = estimate_prompt_tokens("")

        assert tokens == 0

    def test_single_word_returns_at_least_one(self):
        """Single word should return at least one token."""
        tokens = estimate_prompt_tokens("word")

        assert tokens >= 1

    def test_estimate_accounts_for_subword_tokenization(self):
        """Estimate should account for subword tokenization (ratio > 1)."""
        prompt = "word " * 100  # 100 words
        tokens = estimate_prompt_tokens(prompt)

        # With ratio of 1.3, 100 words should yield ~130 tokens
        assert tokens > 100


# =====
# Test PROMISE_EXAMPLES constant
# =====
class TestPromiseExamples:
    """Tests for the PROMISE_EXAMPLES constant."""

    def test_examples_is_list(self):
        """PROMISE_EXAMPLES should be a list."""
        assert isinstance(PROMISE_EXAMPLES, list)

    def test_examples_not_empty(self):
        """PROMISE_EXAMPLES should not be empty."""
        assert len(PROMISE_EXAMPLES) > 0

    def test_examples_are_strings(self):
        """All examples should be strings."""
        for example in PROMISE_EXAMPLES:
            assert isinstance(example, str)

    def test_contains_expected_examples(self):
        """Should contain key examples from analysis_up8.py."""
        expected = ["deal", "yes", "agreed", "lets do it"]

        for exp in expected:
            assert exp in PROMISE_EXAMPLES
