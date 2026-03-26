"""
OpenAI embeddings client for chat message embedding computation.

Provides functions to embed texts using OpenAI's embedding models
with batching support and retry logic for rate limits.

Author: Claude Code
Date: 2026-03-15
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIStatusError

# Load .env from project root
load_dotenv(Path(__file__).parent.parent.parent.parent / '.env')

# MODEL CONFIGURATION
MODEL_SMALL = "text-embedding-3-small"
MODEL_LARGE = "text-embedding-3-large"
MAX_BATCH_SIZE = 2048
MAX_RETRIES = 3
BASE_DELAY_SECONDS = 1.0

# COST ESTIMATES (per 1K tokens)
COST_PER_1K = {
    MODEL_SMALL: 0.00002,
    MODEL_LARGE: 0.00013,
}

# EMBEDDING DIMENSIONS
DIMENSIONS = {
    MODEL_SMALL: 1536,
    MODEL_LARGE: 3072,
}


# =====
# Main embedding functions
# =====
def embed_texts(
    texts: list[str], model: str = MODEL_SMALL
) -> list[list[float]]:
    """Embed a list of texts, auto-batching when exceeding MAX_BATCH_SIZE."""
    if not texts:
        return []

    client = _get_client()
    batches = _split_into_batches(texts)
    all_embeddings = []

    for batch in batches:
        embeddings = _embed_batch_with_retry(client, batch, model)
        all_embeddings.extend(embeddings)

    return all_embeddings


def embed_single(
    text: str, model: str = MODEL_SMALL
) -> list[float]:
    """Embed a single text string. Convenience wrapper around embed_texts."""
    result = embed_texts([text], model=model)
    return result[0]


def get_embedding_cost_estimate(
    num_texts: int, avg_tokens: int, model: str = MODEL_SMALL
) -> dict:
    """Estimate API costs for embedding num_texts with avg_tokens each."""
    total_tokens = num_texts * avg_tokens
    if model not in COST_PER_1K:
        raise ValueError(f"Unknown model '{model}'. Valid: {list(COST_PER_1K.keys())}")
    cost_rate = COST_PER_1K[model]
    cost = (total_tokens / 1000) * cost_rate

    return {
        'total_tokens': total_tokens,
        'estimated_cost_usd': round(cost, 6),
        'model': model,
    }


# =====
# Client initialization
# =====
def _get_client() -> OpenAI:
    """Create OpenAI client, validating API key exists."""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


# =====
# Batching
# =====
def _split_into_batches(texts: list[str]) -> list[list[str]]:
    """Split texts into batches of MAX_BATCH_SIZE."""
    return [
        texts[i:i + MAX_BATCH_SIZE]
        for i in range(0, len(texts), MAX_BATCH_SIZE)
    ]


# =====
# API call with retry logic
# =====
def _embed_batch_with_retry(
    client: OpenAI, texts: list[str], model: str
) -> list[list[float]]:
    """Embed a single batch with exponential backoff retry."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return _make_embed_call(client, texts, model)
        except RateLimitError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                _wait_with_backoff(attempt)
        except APIStatusError as e:
            if e.status_code in (500, 502, 503):
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    _wait_with_backoff(attempt)
            else:
                raise
    raise RuntimeError(
        f"Embedding failed after {MAX_RETRIES} retries: {last_error}"
    )


def _make_embed_call(
    client: OpenAI, texts: list[str], model: str
) -> list[list[float]]:
    """Execute single embedding API call."""
    response = client.embeddings.create(
        input=texts,
        model=model,
    )
    return [item.embedding for item in response.data]


def _wait_with_backoff(attempt: int) -> None:
    """Wait with exponential backoff before retry."""
    delay = BASE_DELAY_SECONDS * (2 ** attempt)
    time.sleep(delay)
