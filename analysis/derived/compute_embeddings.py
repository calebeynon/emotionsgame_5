"""
Compute text embeddings for chat messages using OpenAI embedding models.

Reads promise classifications data, explodes JSON messages into individual rows,
merges player state labels, and computes embeddings via the OpenAI API.
Outputs parquet files with metadata and embedding dimensions.

Author: Claude Code
Date: 2026-03-15
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm_clients.embedding_client import embed_texts

# FILE PATHS
INPUT_FILE = (
    Path(__file__).parent.parent / 'datastore' / 'derived'
    / 'promise_classifications.csv'
)
STATE_FILE = (
    Path(__file__).parent.parent / 'datastore' / 'derived'
    / 'player_state_classification.csv'
)
OUTPUT_DIR = Path(__file__).parent.parent / 'datastore' / 'derived'
OUTPUT_SMALL = OUTPUT_DIR / 'embeddings_small.parquet'
OUTPUT_LARGE = OUTPUT_DIR / 'embeddings_large.parquet'
OUTPUT_PR_SMALL = OUTPUT_DIR / 'embeddings_player_round_small.parquet'
OUTPUT_PR_LARGE = OUTPUT_DIR / 'embeddings_player_round_large.parquet'

# EMBEDDING MODELS
MODEL_SMALL = 'text-embedding-3-small'
MODEL_LARGE = 'text-embedding-3-large'

# ID columns carried through to output
ID_COLS = [
    'session_code', 'treatment', 'segment', 'round',
    'group', 'label', 'message_index', 'message_text',
]
PR_ID_COLS = [
    'session_code', 'treatment', 'segment', 'round',
    'group', 'label', 'combined_text',
]


# =====
# Main function
# =====
def main():
    """Main execution flow."""
    messages_df = load_messages()
    print(f"Loaded {len(messages_df)} individual messages")

    state_df = load_state_data()
    messages_df = merge_state_labels(messages_df, state_df)
    print(f"Merged state labels ({messages_df['player_state'].notna().sum()} matched)")

    _compute_model(messages_df, MODEL_SMALL, OUTPUT_SMALL, ID_COLS)
    _compute_model(messages_df, MODEL_LARGE, OUTPUT_LARGE, ID_COLS)

    pr_df = aggregate_player_round(messages_df)
    print(f"\nAggregated to {len(pr_df)} player-round texts")

    _compute_model(pr_df, MODEL_SMALL, OUTPUT_PR_SMALL, PR_ID_COLS)
    _compute_model(pr_df, MODEL_LARGE, OUTPUT_PR_LARGE, PR_ID_COLS)


# =====
# Data loading
# =====
def load_messages() -> pd.DataFrame:
    """Load promise classifications and explode JSON messages into rows."""
    df = pd.read_csv(INPUT_FILE)
    records = []

    for row_idx, row in df.iterrows():
        messages = _parse_messages_json(row, row_idx)
        for idx, text in enumerate(messages):
            records.append(_build_message_record(row, idx, text))

    return pd.DataFrame.from_records(records)


def _parse_messages_json(row: pd.Series, row_idx: int) -> list:
    """Parse JSON messages column, re-raising with row context on failure."""
    try:
        return json.loads(row['messages'])
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(
            f"Malformed JSON at row {row_idx} "
            f"(label={row.get('label')}, segment={row.get('segment')}, "
            f"round={row.get('round')}): {exc.msg}",
            exc.doc,
            exc.pos,
        ) from exc


def _build_message_record(row: pd.Series, idx: int, text: str) -> dict:
    """Build a single message record from a player-round row."""
    return {
        'session_code': row['session_code'],
        'treatment': row['treatment'],
        'segment': row['segment'],
        'round': row['round'],
        'group': row['group'],
        'label': row['label'],
        'message_index': idx,
        'message_text': text,
    }


def load_state_data() -> pd.DataFrame:
    """Load player state classification data with column renaming."""
    df = pd.read_csv(STATE_FILE)
    return df.rename(columns={
        'round_num': 'round',
        'group_id': 'group',
    })


# =====
# Player-round aggregation
# =====
def aggregate_player_round(messages_df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate all messages per player-round into a single text."""
    group_keys = ['session_code', 'treatment', 'segment', 'round', 'group', 'label']
    agg = messages_df.groupby(group_keys).agg(
        combined_text=('message_text', ' '.join),
        player_state=('player_state', 'first'),
    ).reset_index()
    return agg


# =====
# State merging
# =====
def merge_state_labels(
    messages_df: pd.DataFrame,
    state_df: pd.DataFrame,
) -> pd.DataFrame:
    """LEFT JOIN player_state onto messages by shared keys."""
    join_keys = ['session_code', 'segment', 'round', 'group', 'label']
    state_subset = state_df[join_keys + ['player_state']].copy()

    return messages_df.merge(state_subset, on=join_keys, how='left')


# =====
# Embedding computation
# =====
def _compute_model(
    messages_df: pd.DataFrame,
    model: str,
    output_path: Path,
    id_cols: list[str],
) -> None:
    """Compute embeddings for one model, skipping if cached."""
    if output_path.exists():
        print(f"Cache hit: {output_path.name} already exists, skipping")
        return

    text_col = 'combined_text' if 'combined_text' in messages_df.columns else 'message_text'
    print_cost_estimate(len(messages_df), model)
    texts = messages_df[text_col].tolist()
    embeddings = np.array(embed_texts(texts, model=model))

    result_df = build_output_df(messages_df, embeddings, model, id_cols)
    _save_parquet(result_df, output_path)


def print_cost_estimate(num_messages: int, model: str) -> None:
    """Print estimated API cost before making calls."""
    from llm_clients.embedding_client import get_embedding_cost_estimate

    estimate = get_embedding_cost_estimate(num_messages, avg_tokens=20, model=model)
    print(f"\n{model}: ~{num_messages} messages, ~{estimate['total_tokens']} tokens")
    print(f"  Estimated cost: ${estimate['estimated_cost_usd']:.4f}")


def build_output_df(
    messages_df: pd.DataFrame,
    embeddings: np.ndarray,
    model: str,
    id_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Combine metadata columns with embedding dimensions."""
    cols = id_cols if id_cols is not None else ID_COLS
    meta = messages_df[cols + ['player_state']].reset_index(drop=True)
    meta['model'] = model

    emb_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)

    return pd.concat([meta, emb_df], axis=1)


# =====
# Output
# =====
def _save_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Save DataFrame to parquet file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path.name}")


# %%
if __name__ == "__main__":
    main()
