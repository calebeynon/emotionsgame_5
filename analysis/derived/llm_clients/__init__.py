"""
LLM Clients Package for Promise Classification and Embeddings.

This package provides a unified interface for classifying promises in chat
messages using different LLM providers (OpenAI and Anthropic), and for
computing text embeddings via OpenAI. It includes prompt templates, cost
estimation utilities, and provider-specific clients.

Exports:
    classify_promise_openai: Classify promises using OpenAI API
    classify_promise_anthropic: Classify promises using Anthropic API
    build_classification_prompt: Build the prompt for promise classification
    get_openai_cost_estimate: Estimate cost for OpenAI API calls
    get_anthropic_cost_estimate: Estimate cost for Anthropic API calls
    estimate_prompt_tokens: Estimate token count for a prompt
    embed_texts: Batch embed texts using OpenAI embedding API
    embed_single: Embed a single text string
    get_embedding_cost_estimate: Estimate cost for embedding API calls

Author: Claude Code
Date: 2026-01-16
"""

from .openai_client import classify_promise_openai, get_openai_cost_estimate
from .anthropic_client import classify_promise_anthropic, get_anthropic_cost_estimate
from .prompt_templates import build_classification_prompt, estimate_prompt_tokens
from .embedding_client import embed_texts, embed_single, get_embedding_cost_estimate

__all__ = [
    "classify_promise_openai",
    "classify_promise_anthropic",
    "build_classification_prompt",
    "get_openai_cost_estimate",
    "get_anthropic_cost_estimate",
    "estimate_prompt_tokens",
    "embed_texts",
    "embed_single",
    "get_embedding_cost_estimate",
]
