"""
Prompt templates for promise classification in public goods game experiments.

This module provides functions to build prompts for LLM-based promise classification
and estimate token counts for cost estimation.

Author: Claude Code
Date: 2026-01-16
"""

from typing import List, Dict

# Promise examples from behavioral coding (analysis_up8.py)
PROMISE_EXAMPLES = [
    "ok im in",
    "i'm down",
    "yeah thats fine with me",
    "good with me",
    "yeah lets do 25",
    "lets do it",
    "25 we go again",
    "I'm still doing 25 lol",
    "deal",
    "lets all do 25",
    "ya lets all donate everything possible",
    "lets keep it this way",
    "yeah do it everytime",
    "everyone do all 25",
    "done",
    "yes",
    "yup",
    "yea",
    "lets put 25 in the group account so that we all can earn 4",
    "agreed",
]

# Tokens per word ratio (accounts for subword tokenization)
TOKENS_PER_WORD_RATIO = 1.3


# =====
# Main functions
# =====
def build_classification_prompt(message: str, context: List[Dict[str, str]]) -> str:
    """
    Build a prompt for classifying whether a message contains a promise.

    Args:
        message: The target message to classify
        context: List of prior messages in the conversation, each with 'sender' and 'body' keys

    Returns:
        Formatted prompt string for LLM classification
    """
    context_section = _format_context(context)
    examples_section = _format_examples()

    prompt = f"""{_get_game_explanation()}

{_get_classification_instructions()}

{examples_section}

{_get_edge_case_instructions()}

{context_section}

TARGET MESSAGE TO CLASSIFY:
"{message}"

Respond with only a single digit: 1 if the message is a promise, 0 if it is not."""

    return prompt


def estimate_prompt_tokens(prompt: str) -> int:
    """
    Estimate the number of tokens in a prompt.

    Uses word count multiplied by a ratio to account for subword tokenization.
    This is a rough estimate suitable for cost estimation.

    Args:
        prompt: The prompt text to estimate tokens for

    Returns:
        Estimated token count
    """
    word_count = len(prompt.split())
    return int(word_count * TOKENS_PER_WORD_RATIO)


# =====
# Helper functions for prompt building
# =====
def _get_game_explanation() -> str:
    """Return explanation of the public goods game context."""
    return """CONTEXT: PUBLIC GOODS GAME EXPERIMENT
This is a behavioral economics experiment where 4 players in a group each have 25 points.
Players can contribute any amount (0-25) to a shared group account.
All contributions are multiplied by 1.6 and split equally among group members.
Contributing everything (25) maximizes group earnings, but individuals can free-ride
by contributing nothing while others contribute."""


def _get_classification_instructions() -> str:
    """Return core classification instructions."""
    return """TASK: PROMISE CLASSIFICATION
Classify whether the target message contains a promise to contribute to the group account.
A promise includes:
- Explicit commitments to contribute (e.g., "I'll put in 25")
- Agreements to proposals (e.g., "sounds good", "I'm in")
- Expressions of intent to cooperate (e.g., "let's all do it")"""


def _format_examples() -> str:
    """Format promise examples as a bulleted list."""
    examples_text = "\n".join([f'- "{example}"' for example in PROMISE_EXAMPLES])
    return f"""EXAMPLES OF PROMISES:
{examples_text}"""


def _get_edge_case_instructions() -> str:
    """Return instructions for handling edge cases."""
    return """IMPORTANT - WHAT IS A PROMISE (classify as 1):
- Proposals to contribute: "lets all do 25", "everyone put in 25", "I'll do 25"
- Agreements/affirmations responding to a proposal: "okay", "yes", "sounds good", "I'm in", "deal"
- Explicit commitments: "I promise to put 25", "I'll contribute everything"

IMPORTANT - WHAT IS NOT A PROMISE (classify as 0):
- Questions with question marks: "lets all do 25?" or "should we do 25?"
- Statements about current state: "we are in order lol", "we're doing well"
- Encouragements without commitment: "we should earn max guys", "come on team"
- Statements about past actions: "I put 20", "I contributed last round"
- Vague suggestions without commitment: "like around 15-20ish range"
- General chat unrelated to contributions"""


def _format_context(context: List[Dict[str, str]]) -> str:
    """
    Format conversation context for the prompt.

    Args:
        context: List of prior messages with 'sender' and 'body' keys

    Returns:
        Formatted context section or note if no context
    """
    if not context:
        return "CONVERSATION CONTEXT:\n(No prior messages in this round)"

    lines = ["CONVERSATION CONTEXT (prior messages in this round):"]
    for msg in context:
        sender = msg.get("sender", "Unknown")
        body = msg.get("body", "")
        lines.append(f"[{sender}]: {body}")

    return "\n".join(lines)
