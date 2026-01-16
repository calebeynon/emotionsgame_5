"""
Anthropic API client for promise classification in chat messages.

Provides functions to classify messages as promises using Claude models
with context from prior conversation.

Author: Claude Code
Date: 2026-01-16
"""

import os
import time
from typing import Any

import anthropic

from .prompt_templates import build_classification_prompt

# MODEL CONFIGURATION
MODEL_NAME = "claude-haiku-4-5-20251001"
MAX_RETRIES = 3
BASE_DELAY_SECONDS = 1.0

# COST ESTIMATES (per 1M tokens for Haiku 4.5)
INPUT_COST_PER_M = 0.80
OUTPUT_COST_PER_M = 4.00
AVG_OUTPUT_TOKENS = 10


# =====
# Main classification function
# =====
def classify_promise_anthropic(message: str, context: list[dict]) -> dict[str, Any]:
    """
    Classify if a message is a promise using Anthropic API.

    Args:
        message: The text to classify
        context: List of prior message dicts with 'sender' and 'body' keys

    Returns:
        Dict with 'classification' (0 or 1) and 'raw_response' (str)

    Raises:
        ValueError: If ANTHROPIC_API_KEY not set
    """
    client = _get_client()
    prompt = _build_prompt(message, context)

    return _call_api_with_retry(client, prompt)


def get_anthropic_cost_estimate(num_messages: int, avg_context_length: int) -> dict:
    """
    Estimate API costs for batch classification.

    Args:
        num_messages: Number of messages to classify
        avg_context_length: Average number of context messages per call

    Returns:
        Dict with 'input_tokens', 'output_tokens', 'estimated_cost_usd'
    """
    tokens_per_context_msg = 15
    base_prompt_tokens = 400  # Larger due to detailed prompt
    message_tokens = 20

    input_tokens = num_messages * (
        base_prompt_tokens + message_tokens +
        (avg_context_length * tokens_per_context_msg)
    )
    output_tokens = num_messages * AVG_OUTPUT_TOKENS

    input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_M
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_M

    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'estimated_cost_usd': round(input_cost + output_cost, 4)
    }


# =====
# Client initialization
# =====
def _get_client() -> anthropic.Anthropic:
    """Create Anthropic client, validating API key exists."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    return anthropic.Anthropic(api_key=api_key)


# =====
# Prompt construction
# =====
def _build_prompt(message: str, context: list[dict]) -> str:
    """Build classification prompt using shared template."""
    return build_classification_prompt(message, context)


# =====
# API call with retry logic
# =====
def _call_api_with_retry(client: anthropic.Anthropic, prompt: str) -> dict[str, Any]:
    """Call API with exponential backoff retry for rate limits."""
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            return _make_api_call(client, prompt)
        except anthropic.RateLimitError as e:
            last_error = e
            _wait_with_backoff(attempt)
        except anthropic.APIError as e:
            last_error = e
            if attempt == MAX_RETRIES - 1:
                break
            _wait_with_backoff(attempt)

    return _handle_failure(last_error)


def _make_api_call(client: anthropic.Anthropic, prompt: str) -> dict[str, Any]:
    """Execute single API call and parse response."""
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    classification = _parse_response(raw)

    return {'classification': classification, 'raw_response': raw}


def _parse_response(raw: str) -> int:
    """Parse API response to get 0 or 1 classification."""
    if '1' in raw:
        return 1
    return 0


def _wait_with_backoff(attempt: int) -> None:
    """Wait with exponential backoff before retry."""
    delay = BASE_DELAY_SECONDS * (2 ** attempt)
    time.sleep(delay)


def _handle_failure(error: Exception) -> dict[str, Any]:
    """Return failure result after all retries exhausted."""
    return {
        'classification': 0,
        'raw_response': f"Error: {str(error)}"
    }
