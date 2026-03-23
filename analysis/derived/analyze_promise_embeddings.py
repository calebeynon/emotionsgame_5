"""
Analyze chat embeddings along the promise vs no-promise axis.

Computes promise/no-promise direction vectors from embeddings,
projects all messages onto this axis, validates with probe phrases,
and outputs projection scores for downstream analysis.

Author: Claude Code
Date: 2026-03-15
"""

from pathlib import Path

import numpy as np
import pandas as pd

from analyze_embeddings import (
    load_embeddings,
    compute_difference_vector,
    project_onto_direction,
    cosine_similarities,
)
from llm_clients.embedding_client import embed_texts, MODEL_SMALL, MODEL_LARGE

# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / 'datastore' / 'derived'
EMBEDDINGS_SMALL = DERIVED_DIR / 'embeddings_small.parquet'
EMBEDDINGS_LARGE = DERIVED_DIR / 'embeddings_large.parquet'
EMBEDDINGS_PR_SMALL = DERIVED_DIR / 'embeddings_player_round_small.parquet'
EMBEDDINGS_PR_LARGE = DERIVED_DIR / 'embeddings_player_round_large.parquet'
STATE_FILE = DERIVED_DIR / 'player_state_classification.csv'
PROJECTIONS_OUTPUT = DERIVED_DIR / 'promise_embedding_projections.csv'
PROJECTIONS_PR_OUTPUT = (
    DERIVED_DIR / 'promise_embedding_projections_player_round.csv'
)

# ANALYSIS CONSTANTS
STATE_PROMISE = 'promise'
STATE_NO_PROMISE = 'no_promise'
PROMISE_COL = 'promise_label'

PROBE_PHRASES = [
    "I promise to put in 25",
    "I'll contribute the full amount",
    "I'm not making any promises",
    "I'll do whatever I want",
    "we all agreed to cooperate",
    "I said I would contribute",
    "I never said I'd do anything",
    "let's all commit to the max",
    "I swear I'll put in everything",
    "no guarantees from me",
    "trust me I'll do my part",
    "I might change my mind",
]

ID_COLS = [
    'session_code', 'treatment', 'segment', 'round',
    'group', 'label', 'message_index', 'message_text',
]
PR_ID_COLS = [
    'session_code', 'treatment', 'segment', 'round',
    'group', 'label', 'combined_text',
]

JOIN_KEYS = ['session_code', 'segment', 'round', 'group', 'label']


# =====
# Main function
# =====
def main():
    """Main execution flow."""
    promise_df = load_promise_labels()
    pr_dir = _run_player_round_analysis(promise_df)
    _run_message_analysis(pr_dir, promise_df)


def _run_player_round_analysis(promise_df: pd.DataFrame) -> dict:
    """Compute player-round projections and save CSV. Returns direction vectors."""
    print("=== Player-round level analysis ===")
    pr_dir = {}
    pr_small, pr_dir['small'] = _analyze_model_with_direction(
        EMBEDDINGS_PR_SMALL, 'small', PR_ID_COLS, promise_df,
    )
    pr_large, pr_dir['large'] = _analyze_model_with_direction(
        EMBEDDINGS_PR_LARGE, 'large', PR_ID_COLS, promise_df,
    )
    pr_combined = _merge_projections(pr_small, pr_large, PR_ID_COLS)
    pr_combined.to_csv(PROJECTIONS_PR_OUTPUT, index=False)
    print(f"\nSaved {len(pr_combined)} rows to {PROJECTIONS_PR_OUTPUT.name}")
    return pr_dir


def _run_message_analysis(pr_dir: dict, promise_df: pd.DataFrame) -> None:
    """Compute message-level cross-level projections and save CSV."""
    print("\n=== Message-level analysis ===")
    combined = _analyze_messages_cross_level(pr_dir, promise_df)
    combined.to_csv(PROJECTIONS_OUTPUT, index=False)
    print(f"\nSaved {len(combined)} rows to {PROJECTIONS_OUTPUT.name}")


# =====
# Promise label loading
# =====
def load_promise_labels() -> pd.DataFrame:
    """Load made_promise from player_state_classification.csv."""
    df = pd.read_csv(STATE_FILE)
    df = df.rename(columns={'round_num': 'round', 'group_id': 'group'})
    df[PROMISE_COL] = df['made_promise'].map(
        {True: STATE_PROMISE, False: STATE_NO_PROMISE}
    )
    return df[JOIN_KEYS + [PROMISE_COL]]


def merge_promise_labels(
    meta: pd.DataFrame, promise_df: pd.DataFrame
) -> pd.DataFrame:
    """LEFT JOIN promise labels onto embedding metadata."""
    return meta.merge(promise_df, on=JOIN_KEYS, how='left')


# =====
# Centroid computation (promise-based)
# =====
def compute_promise_centroids(
    embeddings: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean embedding for promise vs no-promise groups."""
    promise_mask = labels == STATE_PROMISE
    no_promise_mask = labels == STATE_NO_PROMISE
    if not promise_mask.any():
        raise ValueError(f"No '{STATE_PROMISE}' labels found. Labels: {set(labels)}")
    if not no_promise_mask.any():
        raise ValueError(f"No '{STATE_NO_PROMISE}' labels found. Labels: {set(labels)}")
    promise_centroid = embeddings[promise_mask].mean(axis=0)
    no_promise_centroid = embeddings[no_promise_mask].mean(axis=0)
    return promise_centroid, no_promise_centroid


# =====
# Direction computation
# =====
def _compute_promise_direction(
    meta: pd.DataFrame, embeddings: np.ndarray
) -> np.ndarray:
    """Compute normalized promise direction vector from labeled embeddings."""
    labels = meta[PROMISE_COL].values
    promise_c, no_promise_c = compute_promise_centroids(embeddings, labels)
    direction = compute_difference_vector(promise_c, no_promise_c)

    n_promise = (labels == STATE_PROMISE).sum()
    n_no_promise = (labels == STATE_NO_PROMISE).sum()
    print(f"  Promise: {n_promise}, No-promise: {n_no_promise}")
    return direction


# =====
# Model-level analysis
# =====
def _analyze_model_with_direction(
    path: Path, suffix: str, id_cols: list[str],
    promise_df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Run analysis pipeline, returning projections AND direction vector."""
    print(f"\n--- Analyzing {suffix} model ---")
    meta, embeddings = load_embeddings(path)
    meta = merge_promise_labels(meta, promise_df)

    direction = _compute_promise_direction(meta, embeddings)
    projections = project_onto_direction(embeddings, direction)

    text_col = _get_text_col(meta)
    _print_rankings(meta, projections, text_col)
    _run_probe_validation(direction, suffix)

    proj_df = _build_output(meta, projections, suffix, id_cols)
    return proj_df, direction


def _get_text_col(meta: pd.DataFrame) -> str:
    """Determine text column name from metadata."""
    if 'combined_text' in meta.columns:
        return 'combined_text'
    return 'message_text'


# =====
# Cross-level message analysis
# =====
def _analyze_messages_cross_level(
    pr_directions: dict[str, np.ndarray],
    promise_df: pd.DataFrame,
) -> pd.DataFrame:
    """Project messages onto both message-level and player-round directions."""
    results = {}
    for suffix, msg_path in [('small', EMBEDDINGS_SMALL), ('large', EMBEDDINGS_LARGE)]:
        print(f"\n--- Analyzing {suffix} model ---")
        meta, embeddings = load_embeddings(msg_path)
        meta = merge_promise_labels(meta, promise_df)

        msg_dir = _compute_promise_direction(meta, embeddings)
        msg_proj = project_onto_direction(embeddings, msg_dir)
        pr_proj = project_onto_direction(embeddings, pr_directions[suffix])
        results[suffix] = (meta, msg_proj, pr_proj)

    return _build_cross_level_output(results)


def _build_cross_level_output(
    results: dict,
) -> pd.DataFrame:
    """Build combined output with both projection types."""
    meta = results['small'][0]
    out = meta[ID_COLS + ['player_state', PROMISE_COL]].copy()
    out['proj_promise_msg_dir_small'] = results['small'][1]
    out['proj_promise_pr_dir_small'] = results['small'][2]
    out['proj_promise_msg_dir_large'] = results['large'][1]
    out['proj_promise_pr_dir_large'] = results['large'][2]
    return out


# =====
# Output construction
# =====
def _build_output(
    meta: pd.DataFrame, projections: np.ndarray,
    suffix: str, id_cols: list[str],
) -> pd.DataFrame:
    """Build output DataFrame with ID columns and projection score."""
    out = meta[id_cols + ['player_state', PROMISE_COL]].copy()
    out[f'proj_promise_pr_dir_{suffix}'] = projections
    return out


def _merge_projections(
    proj_small: pd.DataFrame, proj_large: pd.DataFrame,
    id_cols: list[str],
) -> pd.DataFrame:
    """Merge small and large projection scores on ID columns."""
    merge_keys = id_cols + ['player_state', PROMISE_COL]
    large_cols = merge_keys + ['proj_promise_pr_dir_large']
    return proj_small.merge(proj_large[large_cols], on=merge_keys)


# =====
# Ranking
# =====
def _print_rankings(
    meta: pd.DataFrame, projections: np.ndarray,
    text_col: str = 'message_text',
) -> None:
    """Print top promise and no-promise messages by projection."""
    df = meta[[text_col, PROMISE_COL]].copy()
    df['projection'] = projections
    sorted_df = df.sort_values('projection', ascending=False)

    print("\n  Top promise-direction texts:")
    for _, row in sorted_df.head(5).iterrows():
        print(f"    [{row['projection']:.4f}] {row[text_col][:60]}")

    print("\n  Top no-promise-direction texts:")
    for _, row in sorted_df.tail(5).iterrows():
        print(f"    [{row['projection']:.4f}] {row[text_col][:60]}")


# =====
# Probe phrase validation
# =====
def probe_phrase_validation(
    direction: np.ndarray, model: str
) -> pd.DataFrame:
    """Embed probe phrases and compute cosine similarity with direction."""
    probe_embeddings = np.array(embed_texts(PROBE_PHRASES, model=model))
    similarities = cosine_similarities(probe_embeddings, direction)

    return pd.DataFrame({
        'phrase': PROBE_PHRASES,
        'cosine_similarity': similarities,
    }).sort_values('cosine_similarity', ascending=False)


def _run_probe_validation(direction: np.ndarray, suffix: str) -> None:
    """Run and print probe phrase validation results."""
    model = MODEL_SMALL if suffix == 'small' else MODEL_LARGE
    probe_df = probe_phrase_validation(direction, model)
    print(f"\n  Probe phrase validation ({suffix}):")
    for _, row in probe_df.iterrows():
        print(f"    [{row['cosine_similarity']:.4f}] {row['phrase']}")


# %%
if __name__ == "__main__":
    main()
