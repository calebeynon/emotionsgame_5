"""
LLM Clients Package for Promise Classification.

This package provides a unified interface for classifying promises in chat
messages using different LLM providers (OpenAI and Anthropic). It includes
prompt templates, cost estimation utilities, and provider-specific clients.

Exports:
    classify_promise_openai: Classify promises using OpenAI API
    classify_promise_anthropic: Classify promises using Anthropic API
    build_classification_prompt: Build the prompt for promise classification
    get_openai_cost_estimate: Estimate cost for OpenAI API calls
    get_anthropic_cost_estimate: Estimate cost for Anthropic API calls
    estimate_prompt_tokens: Estimate token count for a prompt

Author: Claude Code
Date: 2026-01-16
"""

from .openai_client import classify_promise_openai, get_openai_cost_estimate
from .anthropic_client import classify_promise_anthropic, get_anthropic_cost_estimate
from .prompt_templates import build_classification_prompt, estimate_prompt_tokens

__all__ = [
    "classify_promise_openai",
    "classify_promise_anthropic",
    "build_classification_prompt",
    "get_openai_cost_estimate",
    "get_anthropic_cost_estimate",
    "estimate_prompt_tokens",
]
