"""
Tests for classify_guilt.py pure functions.

Tests the parsing, validation, and prompt-building helpers used by
the liar communication strategy classifier.

Author: Claude Code
Date: 2026-03-23
"""

import json
import math
import sys
from pathlib import Path

import pytest

# Add derived directory to path (where classify_guilt.py lives)
sys.path.insert(0, str(Path(__file__).parent.parent / "derived"))

from classify_guilt import (
    VALID_CATEGORIES,
    _parse_msgs,
    _strip_markdown,
    _validate_categories,
    build_guilt_prompt,
    parse_response,
)


# =====
# Test _strip_markdown
# =====
class TestStripMarkdown:
    """Tests for stripping markdown code fences from model responses."""

    def test_strips_json_code_fence(self):
        """Fenced JSON block should have fences removed."""
        raw = '```json\n{"categories": ["no_guilt"]}\n```'
        assert _strip_markdown(raw) == '{"categories": ["no_guilt"]}'

    def test_strips_plain_code_fence(self):
        """Fenced block without language tag should have fences removed."""
        raw = '```\n{"categories": ["no_guilt"]}\n```'
        assert _strip_markdown(raw) == '{"categories": ["no_guilt"]}'

    def test_no_fences_unchanged(self):
        """Plain JSON string should pass through unchanged."""
        raw = '{"categories": ["no_guilt"]}'
        assert _strip_markdown(raw) == raw

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert _strip_markdown("") == ""

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace should be stripped."""
        raw = '  \n{"categories": ["no_guilt"]}\n  '
        assert _strip_markdown(raw) == '{"categories": ["no_guilt"]}'

    def test_backticks_only_no_newline(self):
        """Edge case: backticks with no newline inside."""
        raw = '```content```'
        assert _strip_markdown(raw) == "content"


# =====
# Test _validate_categories
# =====
class TestValidateCategories:
    """Tests for category validation and contradiction resolution."""

    def test_valid_categories_pass_through(self):
        """Valid categories should be returned unchanged."""
        cats = ["genuine_guilt", "false_promise"]
        assert _validate_categories(cats) == cats

    def test_invalid_categories_filtered(self):
        """Invalid category names should be removed."""
        cats = ["genuine_guilt", "made_up_category", "false_promise"]
        assert _validate_categories(cats) == ["genuine_guilt", "false_promise"]

    def test_empty_list_returns_no_guilt(self):
        """Empty category list should default to no_guilt."""
        assert _validate_categories([]) == ["no_guilt"]

    def test_all_invalid_returns_no_guilt(self):
        """All-invalid list should default to no_guilt."""
        assert _validate_categories(["fake", "bogus"]) == ["no_guilt"]

    def test_no_guilt_with_others_removes_no_guilt(self):
        """no_guilt combined with real categories is contradictory; remove no_guilt."""
        cats = ["false_promise", "no_guilt", "blame_shifting"]
        result = _validate_categories(cats)
        assert "no_guilt" not in result
        assert result == ["false_promise", "blame_shifting"]

    def test_no_guilt_alone_preserved(self):
        """Sole no_guilt should be kept."""
        assert _validate_categories(["no_guilt"]) == ["no_guilt"]

    def test_single_valid_category(self):
        """Single valid category should pass through."""
        assert _validate_categories(["manipulation"]) == ["manipulation"]


# =====
# Test parse_response
# =====
class TestParseResponse:
    """Tests for parsing raw model JSON responses."""

    def test_valid_json(self):
        """Well-formed JSON should parse correctly."""
        raw = '{"categories": ["false_promise", "blame_shifting"], "reasoning": "test"}'
        result = parse_response(raw)
        assert result["categories"] == ["false_promise", "blame_shifting"]
        assert result["reasoning"] == "test"
        assert result["raw"] == raw

    def test_markdown_wrapped_json(self):
        """JSON inside markdown fences should parse correctly."""
        inner = '{"categories": ["genuine_guilt"], "reasoning": "felt bad"}'
        raw = f"```json\n{inner}\n```"
        result = parse_response(raw)
        assert result["categories"] == ["genuine_guilt"]
        assert result["reasoning"] == "felt bad"

    def test_invalid_json_returns_parse_error(self):
        """Non-JSON response should produce parse_error."""
        raw = "This is not valid JSON at all"
        result = parse_response(raw)
        assert result["categories"] == ["parse_error"]
        assert result["raw"] == raw

    def test_missing_categories_key_returns_no_guilt(self):
        """JSON without 'categories' key should default to no_guilt."""
        raw = '{"reasoning": "no categories provided"}'
        result = parse_response(raw)
        assert result["categories"] == ["no_guilt"]

    def test_contradictory_categories_resolved(self):
        """no_guilt + other categories should drop no_guilt."""
        raw = '{"categories": ["false_promise", "no_guilt"], "reasoning": "mixed"}'
        result = parse_response(raw)
        assert result["categories"] == ["false_promise"]
        assert "no_guilt" not in result["categories"]

    def test_missing_reasoning_defaults_empty(self):
        """Missing reasoning key should default to empty string."""
        raw = '{"categories": ["no_guilt"]}'
        result = parse_response(raw)
        assert result["reasoning"] == ""

    def test_all_invalid_categories_defaults_no_guilt(self):
        """All unrecognized categories should default to no_guilt."""
        raw = '{"categories": ["imaginary"], "reasoning": "bad model"}'
        result = parse_response(raw)
        assert result["categories"] == ["no_guilt"]


# =====
# Test build_guilt_prompt
# =====
class TestBuildGuiltPrompt:
    """Tests for user prompt construction."""

    def test_contains_messages(self):
        """Prompt should include each message text."""
        msgs = ["sorry yall", "ill do 25 next time"]
        prompt = build_guilt_prompt(msgs, 5)
        assert "sorry yall" in prompt
        assert "ill do 25 next time" in prompt

    def test_contains_contribution(self):
        """Prompt should state the actual contribution."""
        prompt = build_guilt_prompt(["hello"], 10)
        assert "10" in prompt
        assert "25" in prompt

    def test_contains_valid_categories(self):
        """Prompt should list all valid category names."""
        prompt = build_guilt_prompt(["test"], 0)
        for cat in VALID_CATEGORIES:
            assert cat in prompt

    def test_messages_are_numbered(self):
        """Each message should be numbered sequentially."""
        msgs = ["first", "second", "third"]
        prompt = build_guilt_prompt(msgs, 0)
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "[3]" in prompt

    def test_single_message(self):
        """Single message should still produce valid prompt."""
        prompt = build_guilt_prompt(["only one"], 15)
        assert "only one" in prompt
        assert "[1]" in prompt


# =====
# Test _parse_msgs
# =====
class TestParseMsgs:
    """Tests for JSON message list parsing from DataFrame values."""

    def test_valid_json_list(self):
        """Valid JSON list of strings should parse correctly."""
        val = json.dumps(["hello", "world"])
        assert _parse_msgs(val) == ["hello", "world"]

    def test_nan_returns_empty(self):
        """NaN (missing data) should return empty list."""
        assert _parse_msgs(float("nan")) == []

    def test_none_returns_empty(self):
        """None should return empty list (pd.isna(None) is True)."""
        assert _parse_msgs(None) == []

    def test_malformed_json_returns_empty(self):
        """Invalid JSON string should return empty list."""
        assert _parse_msgs("not json [[[") == []

    def test_non_string_number_returns_empty(self):
        """Numeric value that is not NaN should attempt parse and fail gracefully."""
        # json.loads(42) raises TypeError, caught by except (JSONDecodeError, TypeError)
        assert _parse_msgs(42) == []

    def test_empty_json_list(self):
        """Empty JSON list should return empty list."""
        assert _parse_msgs("[]") == []

    def test_single_message(self):
        """Single-element list should parse correctly."""
        val = json.dumps(["all in"])
        assert _parse_msgs(val) == ["all in"]
