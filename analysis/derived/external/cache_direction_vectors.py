"""
Cache all 5 direction vectors as .npy files from our experiment embeddings.

Recomputes cooperative, promise, round-liar, cumulative-liar, and homogeneity
direction vectors from cached parquet embeddings and label CSVs, saving each
as a unit-normalized .npy vector for reuse in external validation.

Author: Claude Code
Date: 2026-03-26
"""

from pathlib import Path

import numpy as np
import pandas as pd

# FILE PATHS
DERIVED_DIR = Path(__file__).resolve().parent.parent.parent / 'datastore' / 'derived'
EMBEDDINGS_PR_SMALL = DERIVED_DIR / 'embeddings_player_round_small.parquet'
STATE_FILE = DERIVED_DIR / 'player_state_classification.csv'
BEHAVIOR_FILE = DERIVED_DIR / 'behavior_classifications.csv'
SENTIMENT_FILE = DERIVED_DIR / 'sentiment_scores.csv'
OUTPUT_DIR = DERIVED_DIR / 'direction_vectors'

# LABEL CONSTANTS
JOIN_KEYS = ['session_code', 'segment', 'round', 'group', 'label']
GROUP_KEYS = ['session_code', 'segment', 'round', 'group']


# =====
# Main function
# =====
def main():
    """Compute and cache all 5 direction vectors."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    meta, embeddings = load_embeddings(EMBEDDINGS_PR_SMALL)
    print(f"Loaded {len(meta)} player-round embeddings, dim={embeddings.shape[1]}")

    vectors = {
        'cooperative': compute_cooperative_direction(meta, embeddings),
        'promise': compute_promise_direction(meta, embeddings),
        'round_liar': compute_round_liar_direction(meta, embeddings),
        'cumulative_liar': compute_cumulative_liar_direction(meta, embeddings),
        'homogeneity': compute_homogeneity_direction(meta, embeddings),
    }

    for name, vec in vectors.items():
        out_path = OUTPUT_DIR / f'{name}.npy'
        np.save(out_path, vec)
        print(f"  Saved {name}: shape={vec.shape}, norm={np.linalg.norm(vec):.6f}")

    print(f"\nAll 5 direction vectors cached in {OUTPUT_DIR}")


# =====
# Embedding loading
# =====
def load_embeddings(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load parquet, separating metadata from embedding columns."""
    df = pd.read_parquet(path)
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    meta = df.drop(columns=emb_cols)
    embeddings = df[emb_cols].values
    return meta, embeddings


# =====
# Shared centroid math
# =====
def compute_centroids(
    embeddings: np.ndarray, mask_pos: np.ndarray, mask_neg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean embedding for positive vs negative label groups."""
    return embeddings[mask_pos].mean(axis=0), embeddings[mask_neg].mean(axis=0)


def normalize_direction(pos_centroid: np.ndarray, neg_centroid: np.ndarray) -> np.ndarray:
    """Unit-normalized difference vector: positive minus negative."""
    diff = pos_centroid - neg_centroid
    norm = np.linalg.norm(diff)
    if norm == 0:
        raise ValueError("Direction vector has zero norm (centroids identical).")
    return diff / norm


def direction_from_labels(
    embeddings: np.ndarray, labels: np.ndarray,
    pos_label: str, neg_label: str, name: str,
) -> np.ndarray:
    """Compute direction vector from binary string labels."""
    mask_pos = labels == pos_label
    mask_neg = labels == neg_label
    n_pos, n_neg = mask_pos.sum(), mask_neg.sum()
    if n_pos == 0:
        raise ValueError(f"{name}: no '{pos_label}' labels found.")
    if n_neg == 0:
        raise ValueError(f"{name}: no '{neg_label}' labels found.")
    print(f"  {name}: {pos_label}={n_pos}, {neg_label}={n_neg}")
    pos_c, neg_c = compute_centroids(embeddings, mask_pos, mask_neg)
    return normalize_direction(pos_c, neg_c)


# =====
# 1. Cooperative direction (player_state from parquet)
# =====
def compute_cooperative_direction(
    meta: pd.DataFrame, embeddings: np.ndarray,
) -> np.ndarray:
    """Cooperative vs noncooperative from player_state column in parquet."""
    labels = meta['player_state'].values
    return direction_from_labels(
        embeddings, labels, 'cooperative', 'noncooperative', 'cooperative',
    )


# =====
# 2. Promise direction (made_promise from player_state_classification.csv)
# =====
def compute_promise_direction(
    meta: pd.DataFrame, embeddings: np.ndarray,
) -> np.ndarray:
    """Promise vs no-promise from made_promise in state classification."""
    state_df = pd.read_csv(STATE_FILE)
    state_df = state_df.rename(columns={'round_num': 'round', 'group_id': 'group'})
    state_df['promise_label'] = state_df['made_promise'].map(
        {True: 'promise', False: 'no_promise'},
    )
    merged = meta.merge(state_df[JOIN_KEYS + ['promise_label']], on=JOIN_KEYS, how='left')
    labels = merged['promise_label'].values
    return direction_from_labels(
        embeddings, labels, 'promise', 'no_promise', 'promise',
    )


# =====
# 3. Round-liar direction (lied_this_round_20 from behavior_classifications.csv)
# =====
def compute_round_liar_direction(
    meta: pd.DataFrame, embeddings: np.ndarray,
) -> np.ndarray:
    """Liar vs non-liar from lied_this_round_20 in behavior classifications."""
    behavior_df = pd.read_csv(BEHAVIOR_FILE)
    behavior_df['round_liar_label'] = behavior_df['lied_this_round_20'].map(
        _bool_to_label,
    )
    merged = meta.merge(
        behavior_df[JOIN_KEYS + ['round_liar_label']], on=JOIN_KEYS, how='left',
    )
    labels = merged['round_liar_label'].values
    return direction_from_labels(
        embeddings, labels, 'liar', 'non_liar', 'round_liar',
    )


# =====
# 4. Cumulative-liar direction (is_liar_20 from behavior_classifications.csv)
# =====
def compute_cumulative_liar_direction(
    meta: pd.DataFrame, embeddings: np.ndarray,
) -> np.ndarray:
    """Liar vs non-liar from is_liar_20 in behavior classifications."""
    behavior_df = pd.read_csv(BEHAVIOR_FILE)
    behavior_df['cumulative_liar_label'] = behavior_df['is_liar_20'].map(
        _bool_to_label,
    )
    merged = meta.merge(
        behavior_df[JOIN_KEYS + ['cumulative_liar_label']], on=JOIN_KEYS, how='left',
    )
    labels = merged['cumulative_liar_label'].values
    return direction_from_labels(
        embeddings, labels, 'liar', 'non_liar', 'cumulative_liar',
    )


# =====
# 5. Homogeneity direction (contribution range from sentiment_scores.csv)
# =====
def compute_homogeneity_direction(
    meta: pd.DataFrame, embeddings: np.ndarray,
) -> np.ndarray:
    """Homogeneous vs heterogeneous from group contribution range <= 1."""
    sentiment_df = pd.read_csv(SENTIMENT_FILE)
    group_range = _compute_group_contribution_range(sentiment_df)
    group_range['homogeneity_label'] = group_range['contribution_range'].map(
        lambda r: 'homogeneous' if r <= 1 else 'heterogeneous',
    )
    merged = meta.merge(
        group_range[GROUP_KEYS + ['homogeneity_label']], on=GROUP_KEYS, how='left',
    )
    labels = merged['homogeneity_label'].values
    return direction_from_labels(
        embeddings, labels, 'homogeneous', 'heterogeneous', 'homogeneity',
    )


# =====
# Helper functions
# =====
def _bool_to_label(x) -> str | None:
    """Map boolean to liar/non_liar string, preserving NaN."""
    if pd.isna(x):
        return None
    return 'liar' if x else 'non_liar'


def _compute_group_contribution_range(df: pd.DataFrame) -> pd.DataFrame:
    """Max - min contribution per group-round."""
    agg = df.groupby(GROUP_KEYS)['contribution'].agg(['max', 'min'])
    agg['contribution_range'] = agg['max'] - agg['min']
    return agg.reset_index()


# %%
if __name__ == "__main__":
    main()
