"""
Compute OpenAI embeddings for CCR group-level chat transcripts.

Loads cleaned group chat data (116 color teams), embeds non-empty
text using text-embedding-3-small, and saves metadata + emb_0..emb_1535
columns as parquet.

Author: Claude Code
Date: 2026-03-26
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_clients.embedding_client import (
    DIMENSIONS,
    MODEL_SMALL,
    embed_texts,
    get_embedding_cost_estimate,
)

logger = logging.getLogger(__name__)

# FILE PATHS
INPUT_FILE = Path("analysis/datastore/derived/external/ccr_chat_clean.parquet")
OUTPUT_FILE = Path("analysis/datastore/derived/external/ccr_embeddings_small.parquet")

# METADATA COLUMNS to carry through to output
META_COLS = [
    "session", "red", "run", "ingroup", "commonknow",
    "group_chat_text", "n_messages", "n_words", "n_characters",
]


# =====
# Main function
# =====
def main():
    """Embed CCR group chat texts and save with metadata."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = _load_and_filter()
    _log_cost_estimate(len(df))
    embeddings = _compute_embeddings(df["group_chat_text"].tolist())
    result = _build_output(df, embeddings)
    _save_output(result)


# =====
# Data loading
# =====
def _load_and_filter():
    """Load input parquet and filter to rows with non-empty chat text."""
    df = pd.read_parquet(INPUT_FILE)
    mask = df["group_chat_text"].str.strip().astype(bool)
    n_empty = (~mask).sum()
    if n_empty > 0:
        logger.info("Skipping %d groups with empty chat text", n_empty)
    df_chat = df[mask].reset_index(drop=True)
    logger.info("Embedding %d / %d groups", len(df_chat), len(df))
    return df_chat


# =====
# Embedding computation
# =====
def _compute_embeddings(texts):
    """Embed texts via OpenAI and return as numpy array."""
    logger.info("Calling OpenAI %s for %d texts...", MODEL_SMALL, len(texts))
    raw = embed_texts(texts, model=MODEL_SMALL)
    embeddings = np.array(raw)
    expected_dim = DIMENSIONS[MODEL_SMALL]
    if embeddings.shape != (len(texts), expected_dim):
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}, "
            f"expected ({len(texts)}, {expected_dim})"
        )
    return embeddings


def _build_output(df_chat, embeddings):
    """Combine metadata columns with embedding columns."""
    emb_dim = DIMENSIONS[MODEL_SMALL]
    emb_cols = [f"emb_{i}" for i in range(emb_dim)]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    result = pd.concat(
        [df_chat[META_COLS].reset_index(drop=True), emb_df],
        axis=1,
    )
    # Add group_key for merging with effort data
    result["group_key"] = (
        result["session"].astype(int).astype(str)
        + "_"
        + result["red"].astype(int).astype(str)
    )
    return result


# =====
# Output and logging
# =====
def _save_output(result):
    """Write embedding parquet to disk."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_FILE, index=False)
    logger.info("Wrote %d rows x %d cols to %s", *result.shape, OUTPUT_FILE)


def _log_cost_estimate(num_texts):
    """Log estimated API cost before making calls."""
    est = get_embedding_cost_estimate(num_texts, avg_tokens=100)
    logger.info(
        "Cost estimate: ~%d tokens, ~$%.4f",
        est["total_tokens"],
        est["estimated_cost_usd"],
    )


# %%
if __name__ == "__main__":
    main()
