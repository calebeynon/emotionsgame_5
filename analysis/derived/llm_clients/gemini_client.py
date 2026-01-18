"""
Gemini API client for promise classification in chat messages.

Provides functions to classify promises using Google's Gemini API with
retry logic for rate limits and cost estimation utilities.

Author: Claude Code
Date: 2026-01-16
"""

import os
import time
from typing import Optional

import google.generativeai as genai

from .prompt_templates import build_classification_prompt

# API CONFIGURATION
DEFAULT_MODEL = "gemini-2.5-flash"
MAX_RETRIES = 3
BASE_DELAY_SECONDS = 1.0

# Cost estimates (per 1M tokens as of 2025)
GEMINI_INPUT_COST_PER_M = 0.075  # $0.075 per 1M input tokens
GEMINI_OUTPUT_COST_PER_M = 0.30  # $0.30 per 1M output tokens
AVG_TOKENS_PER_WORD = 1.3


# =====
# Main classification function
# =====
def classify_promise_gemini(
    message: str,
    context: Optional[list] = None,
) -> dict:
    """
    Classify if a message is a promise using Gemini API.

    Args:
        message: The message text to classify
        context: Optional list of prior messages with 'sender' and 'body' keys

    Returns:
        Dict with 'classification' (0 or 1) and 'raw_response' (str)

    Raises:
        ValueError: If GEMINI_API_KEY environment variable is not set
    """
    _validate_api_key()
    model = _get_model()
    prompt = _build_prompt(message, context)

    return _classify_with_retry(model, prompt)


# =====
# Cost estimation
# =====
def get_gemini_cost_estimate(num_messages: int, avg_context_length: int = 5) -> dict:
    """
    Estimate cost for classifying messages with Gemini API.

    Args:
        num_messages: Number of messages to classify
        avg_context_length: Average number of context messages per call

    Returns:
        Dict with 'input_tokens', 'output_tokens', 'estimated_cost_usd'
    """
    # Estimate tokens per message classification
    prompt_base_words = 50  # Base prompt instructions
    message_words = 15  # Average message length
    context_words = avg_context_length * 20  # ~20 words per context message

    input_words = prompt_base_words + message_words + context_words
    input_tokens_per_call = int(input_words * AVG_TOKENS_PER_WORD)
    output_tokens_per_call = 5  # Just "0" or "1" response

    total_input = input_tokens_per_call * num_messages
    total_output = output_tokens_per_call * num_messages

    input_cost = (total_input / 1_000_000) * GEMINI_INPUT_COST_PER_M
    output_cost = (total_output / 1_000_000) * GEMINI_OUTPUT_COST_PER_M

    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "estimated_cost_usd": round(input_cost + output_cost, 4),
    }


# =====
# Helper functions
# =====
def _validate_api_key() -> None:
    """Validate that GEMINI_API_KEY is set."""
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable not set")


def _get_model():
    """Configure and return Gemini model instance."""
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    return genai.GenerativeModel(DEFAULT_MODEL)


def _build_prompt(message: str, context: Optional[list] = None) -> str:
    """Build classification prompt using shared template."""
    return build_classification_prompt(message, context or [])


def _classify_with_retry(model, prompt: str) -> dict:
    """Attempt classification with exponential backoff retry."""
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt)
            raw_response = response.text.strip()
            classification = _parse_classification(raw_response)

            return {"classification": classification, "raw_response": raw_response}

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY_SECONDS * (2 ** attempt)
                time.sleep(delay)

    # All retries exhausted
    return {"classification": 0, "raw_response": f"Error: {last_error}"}


def _parse_classification(response_text: str) -> int:
    """Parse LLM response to extract 0 or 1 classification."""
    if response_text.strip() == "1":
        return 1
    return 0
