"""
Cache the 5 direction vectors as .npy files for external validation.

Loads the player-round small embeddings parquet, computes centroid-based
direction vectors for each behavioral axis, and saves as .npy files.

Author: Claude Code
Date: 2026-03-26
"""

from pathlib import Path

import numpy as np
import pandas as pd

# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / 'datastore' / 'derived'
EMBEDDINGS_PR_SMALL = DERIVED_DIR / 'embeddings_player_round_small.parquet'
STATE_FILE = DERIVED_DIR / 'player_state_classification.csv'
SENTIMENT_FILE = DERIVED_DIR / 'sentiment_scores.csv'
BEHAVIOR_FILE = DERIVED_DIR / 'behavior_classifications.csv'
OUTPUT_DIR = DERIVED_DIR / 'direction_vectors'

# DIRECTION VECTOR SPECS: (name, label_col, positive_value, negative_value)
PLAYER_JOIN_KEYS = ['session_code', 'segment', 'round', 'group', 'label']
GROUP_JOIN_KEYS = ['session_code', 'segment', 'round', 'group']


# =====
# Main function
# =====
def main():
    """Compute and cache all 5 direction vectors."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta, embeddings = load_embeddings(EMBEDDINGS_PR_SMALL)

    directions = {
        'cooperative': compute_cooperative_direction(meta, embeddings),
        'promise': compute_promise_direction(meta, embeddings),
        'homogeneity': compute_homogeneity_direction(meta, embeddings),
        'round_liar': compute_round_liar_direction(meta, embeddings),
        'cumulative_liar': compute_cumulative_liar_direction(meta, embeddings),
    }

    for name, direction in directions.items():
        save_direction(name, direction)

    print(f"\nAll 5 direction vectors saved to {OUTPUT_DIR}")


# =====
# Data loading
# =====
def load_embeddings(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load parquet, separating metadata from embedding columns."""
    df = pd.read_parquet(path)
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    meta = df.drop(columns=emb_cols)
    embeddings = df[emb_cols].values
    return meta, embeddings


# =====
# Core computation helpers
# =====
def compute_centroid(embeddings: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute mean embedding for a boolean mask."""
    if not mask.any():
        raise ValueError("No True values in mask for centroid computation")
    return embeddings[mask].mean(axis=0)


def compute_normalized_direction(
    embeddings: np.ndarray,
    positive_mask: np.ndarray,
    negative_mask: np.ndarray,
) -> np.ndarray:
    """Compute normalized difference vector: positive centroid minus negative centroid."""
    pos_centroid = compute_centroid(embeddings, positive_mask)
    neg_centroid = compute_centroid(embeddings, negative_mask)
    diff = pos_centroid - neg_centroid
    norm = np.linalg.norm(diff)
    if norm == 0:
        raise ValueError("Direction vector has zero norm (centroids identical)")
    return diff / norm


# =====
# Direction 1: Cooperative (player_state column in parquet)
# =====
def compute_cooperative_direction(
    meta: pd.DataFrame, embeddings: np.ndarray,
) -> np.ndarray:
    """Cooperative minus noncooperative direction from player_state column."""
    labels = meta['player_state'].values
    pos_mask = labels == 'cooperative'
    neg_mask = labels == 'noncooperative'
    direction = compute_normalized_direction(embeddings, pos_mask, neg_mask)
    print(f"Cooperative: {pos_mask.sum()} coop, {neg_mask.sum()} noncoop")
    return direction


# =====
# Direction 2: Promise (from player_state_classification.csv)
# =====
def compute_promise_direction(
    meta: pd.DataFrame, embeddings: np.ndarray,
) -> np.ndarray:
    """Promise minus no-promise direction from made_promise labels."""
    promise_df = _load_promise_labels()
    merged = meta.merge(promise_df, on=PLAYER_JOIN_KEYS, how='left')
    labels = merged['promise_label'].values
    pos_mask = labels == 'promise'
    neg_mask = labels == 'no_promise'
    direction = compute_normalized_direction(embeddings, pos_mask, neg_mask)
    print(f"Promise: {pos_mask.sum()} promise, {neg_mask.sum()} no_promise")
    return direction


def _load_promise_labels() -> pd.DataFrame:
    """Load and remap promise labels from player_state_classification.csv."""
    df = pd.read_csv(STATE_FILE)
    df = df.rename(columns={'round_num': 'round', 'group_id': 'group'})
    df['promise_label'] = df['made_promise'].map(
        {True: 'promise', False: 'no_promise'}
    )
    return df[PLAYER_JOIN_KEYS + ['promise_label']]


# =====
# Direction 3: Homogeneity (from sentiment_scores.csv contribution range)
# =====
def compute_homogeneity_direction(
    meta: pd.DataFrame, embeddings: np.ndarray,
) -> np.ndarray:
    """Homogeneous minus heterogeneous direction from contribution range."""
    homog_df = _compute_homogeneity_labels()
    merged = meta.merge(homog_df, on=GROUP_JOIN_KEYS, how='left')
    labels = merged['homogeneity_label'].values
    pos_mask = labels == 'homogeneous'
    neg_mask = labels == 'heterogeneous'
    direction = compute_normalized_direction(embeddings, pos_mask, neg_mask)
    print(f"Homogeneity: {pos_mask.sum()} homog, {neg_mask.sum()} heterog")
    return direction


def _compute_homogeneity_labels() -> pd.DataFrame:
    """Label group-rounds as homogeneous (contribution range <= 1) or heterogeneous."""
    df = pd.read_csv(SENTIMENT_FILE)
    agg = df.groupby(GROUP_JOIN_KEYS)['contribution'].agg(['max', 'min'])
    agg['range'] = agg['max'] - agg['min']
    agg['homogeneity_label'] = agg['range'].apply(
        lambda r: 'homogeneous' if r <= 1 else 'heterogeneous'
    )
    return agg.reset_index()[GROUP_JOIN_KEYS + ['homogeneity_label']]


# =====
# Direction 4: Round-liar (from behavior_classifications.csv lied_this_round_20)
# =====
def compute_round_liar_direction(
    meta: pd.DataFrame, embeddings: np.ndarray,
) -> np.ndarray:
    """Liar minus non-liar direction from lied_this_round_20."""
    liar_df = _load_round_liar_labels()
    merged = meta.merge(liar_df, on=PLAYER_JOIN_KEYS, how='left')
    labels = merged['round_liar_label'].values
    pos_mask = labels == 'liar'
    neg_mask = labels == 'non_liar'
    direction = compute_normalized_direction(embeddings, pos_mask, neg_mask)
    print(f"Round-liar: {pos_mask.sum()} liar, {neg_mask.sum()} non-liar")
    return direction


def _load_round_liar_labels() -> pd.DataFrame:
    """Load round-liar labels from behavior_classifications.csv."""
    df = pd.read_csv(BEHAVIOR_FILE)
    df['round_liar_label'] = df['lied_this_round_20'].map(
        {True: 'liar', False: 'non_liar'}
    )
    return df[PLAYER_JOIN_KEYS + ['round_liar_label']]


# =====
# Direction 5: Cumulative-liar (from behavior_classifications.csv is_liar_20)
# =====
def compute_cumulative_liar_direction(
    meta: pd.DataFrame, embeddings: np.ndarray,
) -> np.ndarray:
    """Liar minus non-liar direction from is_liar_20."""
    liar_df = _load_cumulative_liar_labels()
    merged = meta.merge(liar_df, on=PLAYER_JOIN_KEYS, how='left')
    labels = merged['cumulative_liar_label'].values
    pos_mask = labels == 'liar'
    neg_mask = labels == 'non_liar'
    direction = compute_normalized_direction(embeddings, pos_mask, neg_mask)
    print(f"Cumulative-liar: {pos_mask.sum()} liar, {neg_mask.sum()} non-liar")
    return direction


def _load_cumulative_liar_labels() -> pd.DataFrame:
    """Load cumulative liar labels from behavior_classifications.csv."""
    df = pd.read_csv(BEHAVIOR_FILE)
    df['cumulative_liar_label'] = df['is_liar_20'].map(
        {True: 'liar', False: 'non_liar'}
    )
    return df[PLAYER_JOIN_KEYS + ['cumulative_liar_label']]


# =====
# Output
# =====
def save_direction(name: str, direction: np.ndarray) -> None:
    """Save a direction vector as a .npy file."""
    path = OUTPUT_DIR / f'{name}.npy'
    np.save(path, direction)
    print(f"  Saved {name}.npy (shape={direction.shape}, norm={np.linalg.norm(direction):.6f})")


# %%
if __name__ == "__main__":
    main()
