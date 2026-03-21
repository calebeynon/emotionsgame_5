"""
Analyze chat embeddings along the round-liar vs non-liar axis.

Loads behavior classifications (lied_this_round_20), computes liar labels,
projects embeddings onto liar direction vector, and outputs projection scores.

Author: Claude Code
Date: 2026-03-21
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
BEHAVIOR_FILE = DERIVED_DIR / 'behavior_classifications.csv'
PROJECTIONS_OUTPUT = DERIVED_DIR / 'round_liar_embedding_projections.csv'

# ANALYSIS CONSTANTS
LIAR_COL = 'round_liar_label'
RLIAR_COL = LIAR_COL
STATE_LIAR = 'liar'
STATE_NON_LIAR = 'non_liar'

PROBE_PHRASES = [
    "I promise to put in 25",
    "I lied about my contribution",
    "trust me",
    "I kept my word",
    "I always do what I say",
    "I changed my mind",
    "I never break a promise",
    "someone is not being honest",
    "they lied to us",
    "I said I'd cooperate but I didn't",
    "I'll put in everything",
]

JOIN_KEYS = ['session_code', 'segment', 'round', 'group', 'label']
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
    liar_df = compute_round_liar_labels()
    pr_dir = _run_player_round_analysis(liar_df)
    _run_message_analysis(pr_dir, liar_df)


def _run_player_round_analysis(liar_df: pd.DataFrame) -> dict:
    """Compute player-round projections and save. Returns direction vectors."""
    print("=== Player-round level analysis ===")
    pr_dir = {}
    pr_small, pr_dir['small'] = _analyze_model_with_direction(
        EMBEDDINGS_PR_SMALL, 'small', PR_ID_COLS, liar_df,
    )
    pr_large, pr_dir['large'] = _analyze_model_with_direction(
        EMBEDDINGS_PR_LARGE, 'large', PR_ID_COLS, liar_df,
    )
    print(f"\nPlayer-round analysis complete: {len(pr_small)} rows")
    return pr_dir


def _run_message_analysis(pr_dir: dict, liar_df: pd.DataFrame) -> None:
    """Compute message-level cross-level projections and save CSV."""
    print("\n=== Message-level analysis ===")
    combined = _analyze_messages_cross_level(pr_dir, liar_df)
    combined.to_csv(PROJECTIONS_OUTPUT, index=False)
    print(f"\nSaved {len(combined)} rows to {PROJECTIONS_OUTPUT.name}")


# =====
# Round-liar label computation
# =====
def compute_round_liar_labels(
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Label each player-round as liar or non_liar from lied_this_round_20."""
    if df is None:
        df = pd.read_csv(BEHAVIOR_FILE)
    df[LIAR_COL] = _assign_labels(df['lied_this_round_20'])
    _print_label_counts(df)
    return df[JOIN_KEYS + [LIAR_COL]]


def _assign_labels(lied_col: pd.Series) -> pd.Series:
    """Map lied_this_round_20 boolean to liar labels."""
    return lied_col.map(
        lambda x: STATE_LIAR if x else STATE_NON_LIAR
    )


def _print_label_counts(df: pd.DataFrame) -> None:
    """Print distribution of round-liar labels."""
    counts = df[LIAR_COL].value_counts()
    print(f"Round-liar labels: {dict(counts)}")


def merge_liar_labels(
    meta: pd.DataFrame, liar_df: pd.DataFrame,
) -> pd.DataFrame:
    """LEFT JOIN round-liar labels onto embedding metadata."""
    return meta.merge(liar_df, on=JOIN_KEYS, how='left')


merge_round_liar_labels = merge_liar_labels


def compute_liar_centroids(
    embeddings: np.ndarray, labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean embedding for liar vs non-liar players."""
    liar_mask = labels == STATE_LIAR
    non_liar_mask = labels == STATE_NON_LIAR
    liar_centroid = embeddings[liar_mask].mean(axis=0)
    non_liar_centroid = embeddings[non_liar_mask].mean(axis=0)
    return liar_centroid, non_liar_centroid


# Aliases for test compatibility
compute_round_liar_centroids = compute_liar_centroids


# =====
# Direction computation
# =====
def _compute_liar_direction(
    meta: pd.DataFrame, embeddings: np.ndarray,
) -> np.ndarray:
    """Compute normalized liar direction vector."""
    labels = meta[LIAR_COL].values
    liar_c, non_liar_c = compute_liar_centroids(embeddings, labels)
    direction = compute_difference_vector(liar_c, non_liar_c)

    n_liar = (labels == STATE_LIAR).sum()
    n_non_liar = (labels == STATE_NON_LIAR).sum()
    print(f"  Liar: {n_liar}, Non-liar: {n_non_liar}")
    return direction


# =====
# Model-level analysis
# =====
def _analyze_model_with_direction(
    path: Path, suffix: str, id_cols: list[str],
    liar_df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Run analysis pipeline, returning projections AND direction vector."""
    print(f"\n--- Analyzing {suffix} model ---")
    meta, embeddings = load_embeddings(path)
    meta = merge_liar_labels(meta, liar_df)

    direction = _compute_liar_direction(meta, embeddings)
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
    liar_df: pd.DataFrame,
) -> pd.DataFrame:
    """Project messages onto both message-level and player-round directions."""
    results = {}
    for suffix, msg_path in [('small', EMBEDDINGS_SMALL), ('large', EMBEDDINGS_LARGE)]:
        print(f"\n--- Analyzing {suffix} model ---")
        meta, embeddings = load_embeddings(msg_path)
        meta = merge_liar_labels(meta, liar_df)

        msg_dir = _compute_liar_direction(meta, embeddings)
        msg_proj = project_onto_direction(embeddings, msg_dir)
        pr_proj = project_onto_direction(embeddings, pr_directions[suffix])
        results[suffix] = (meta, msg_proj, pr_proj)

    return _build_cross_level_output(results)


def _build_cross_level_output(results: dict) -> pd.DataFrame:
    """Build combined output with both projection types."""
    meta = results['small'][0]
    out = meta[ID_COLS + ['player_state', LIAR_COL]].copy()
    out['proj_rliar_msg_dir_small'] = results['small'][1]
    out['proj_rliar_pr_dir_small'] = results['small'][2]
    out['proj_rliar_msg_dir_large'] = results['large'][1]
    out['proj_rliar_pr_dir_large'] = results['large'][2]
    return out


# =====
# Output construction
# =====
def _build_output(
    meta: pd.DataFrame, projections: np.ndarray,
    suffix: str, id_cols: list[str],
) -> pd.DataFrame:
    """Build output DataFrame with ID columns and projection score."""
    out = meta[id_cols + ['player_state', LIAR_COL]].copy()
    out[f'proj_rliar_pr_dir_{suffix}'] = projections
    return out


# =====
# Ranking
# =====
def _print_rankings(
    meta: pd.DataFrame, projections: np.ndarray,
    text_col: str = 'message_text',
) -> None:
    """Print top liar and non-liar messages by projection."""
    df = meta[[text_col, LIAR_COL]].copy()
    df['projection'] = projections
    sorted_df = df.sort_values('projection', ascending=False)

    print("\n  Top liar-direction texts:")
    for _, row in sorted_df.head(5).iterrows():
        print(f"    [{row['projection']:.4f}] {row[text_col][:60]}")

    print("\n  Top non-liar-direction texts:")
    for _, row in sorted_df.tail(5).iterrows():
        print(f"    [{row['projection']:.4f}] {row[text_col][:60]}")


# =====
# Probe phrase validation
# =====
def probe_phrase_validation(
    direction: np.ndarray, model: str,
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
