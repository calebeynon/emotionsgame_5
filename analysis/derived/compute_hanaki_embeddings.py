"""
Compute OpenAI embeddings for Hanaki & Ozkes chat messages.

Loads preprocessed chat-decisions data, embeds non-empty chat text
using text-embedding-3-small, and outputs a parquet with metadata
columns plus emb_0..emb_1535.

Author: Claude Code
Date: 2026-03-26
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from llm_clients.embedding_client import embed_texts, DIMENSIONS, MODEL_SMALL

logger = logging.getLogger(__name__)

# FILE PATHS
INPUT_FILE = Path("analysis/datastore/derived/hanaki_ozkes_chat_decisions.parquet")
OUTPUT_FILE = Path("analysis/datastore/derived/hanaki_ozkes_embeddings.parquet")

# METADATA COLUMNS to carry through to output
META_COLS = [
    "session_file", "period", "player_id", "group",
    "Inv", "OtherInv", "Profit", "Chat", "Comp", "chat_text",
]


# =====
# Main function
# =====
def main():
    """Embed non-empty Hanaki chat texts and save with metadata."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = pd.read_parquet(INPUT_FILE)
    mask = df["chat_text"].str.strip().astype(bool)
    logger.info("Rows with chat: %d / %d total", mask.sum(), len(df))
    df_chat = df[mask].reset_index(drop=True)
    texts = df_chat["chat_text"].tolist()
    _log_cost_estimate(len(texts))
    embeddings = _compute_embeddings(texts)
    result = _build_output(df_chat, embeddings)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_FILE, index=False)
    logger.info("Wrote %d rows to %s", len(result), OUTPUT_FILE)


# =====
# Embedding computation
# =====
def _compute_embeddings(texts):
    """Embed texts and return as numpy array."""
    logger.info("Embedding %d texts with %s...", len(texts), MODEL_SMALL)
    raw = embed_texts(texts, model=MODEL_SMALL)
    return np.array(raw)


def _build_output(df_chat, embeddings):
    """Combine metadata columns with embedding columns."""
    emb_dim = DIMENSIONS[MODEL_SMALL]
    emb_cols = [f"emb_{i}" for i in range(emb_dim)]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    result = pd.concat(
        [df_chat[META_COLS].reset_index(drop=True), emb_df],
        axis=1,
    )
    return result


def _log_cost_estimate(num_texts):
    """Log estimated API cost before making calls."""
    from llm_clients.embedding_client import get_embedding_cost_estimate
    est = get_embedding_cost_estimate(num_texts, avg_tokens=30)
    logger.info(
        "Cost estimate: ~%d tokens, ~$%.4f",
        est["total_tokens"],
        est["estimated_cost_usd"],
    )


# %%
if __name__ == "__main__":
    main()
