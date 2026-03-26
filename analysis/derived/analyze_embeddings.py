"""
Analyze chat message embeddings: centroids, projections, and rankings.

Computes cooperative vs non-cooperative direction vectors from embeddings,
projects all messages onto this axis, validates with probe phrases,
and outputs projection scores for downstream analysis.

Author: Claude Code
Date: 2026-03-15
"""

from pathlib import Path

import numpy as np
import pandas as pd

from llm_clients.embedding_client import embed_texts, MODEL_SMALL, MODEL_LARGE

# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / 'datastore' / 'derived'
EMBEDDINGS_SMALL = DERIVED_DIR / 'embeddings_small.parquet'
EMBEDDINGS_LARGE = DERIVED_DIR / 'embeddings_large.parquet'
EMBEDDINGS_PR_SMALL = DERIVED_DIR / 'embeddings_player_round_small.parquet'
EMBEDDINGS_PR_LARGE = DERIVED_DIR / 'embeddings_player_round_large.parquet'
PROJECTIONS_OUTPUT = DERIVED_DIR / 'embedding_projections.csv'
PROJECTIONS_PR_OUTPUT = DERIVED_DIR / 'embedding_projections_player_round.csv'

# ANALYSIS CONSTANTS
STATE_COOPERATIVE = 'cooperative'
STATE_NONCOOPERATIVE = 'noncooperative'

PROBE_PHRASES = [
    "let's all contribute 25",
    "I'm going to free ride",
    "we should cooperate",
    "I don't trust them",
    "put in everything",
    "I'll put in zero",
]

ID_COLS = [
    'session_code', 'treatment', 'segment', 'round',
    'group', 'label', 'message_index', 'message_text',
]
PR_ID_COLS = [
    'session_code', 'treatment', 'segment', 'round',
    'group', 'label', 'combined_text',
]


# ===== Main function =====
def main():
    """Main execution flow."""
    pr_dir = _run_player_round_analysis()
    _run_message_level_analysis(pr_dir)


def _run_player_round_analysis() -> dict[str, np.ndarray]:
    """Analyze player-round embeddings and save projections. Returns direction vectors."""
    print("=== Player-round level analysis ===")
    pr_dir = {}
    pr_small, pr_dir['small'] = _analyze_model_with_direction(
        EMBEDDINGS_PR_SMALL, 'small', PR_ID_COLS,
    )
    pr_large, pr_dir['large'] = _analyze_model_with_direction(
        EMBEDDINGS_PR_LARGE, 'large', PR_ID_COLS,
    )
    pr_combined = _merge_projections(pr_small, pr_large, PR_ID_COLS)
    pr_combined.to_csv(PROJECTIONS_PR_OUTPUT, index=False)
    print(f"\nSaved {len(pr_combined)} rows to {PROJECTIONS_PR_OUTPUT.name}")
    return pr_dir


def _run_message_level_analysis(pr_dir: dict[str, np.ndarray]) -> None:
    """Project messages onto both message-level and player-round directions."""
    print("\n=== Message-level analysis (own direction + player-round direction) ===")
    combined = _analyze_messages_cross_level(pr_dir)
    combined.to_csv(PROJECTIONS_OUTPUT, index=False)
    print(f"\nSaved {len(combined)} rows to {PROJECTIONS_OUTPUT.name}")


def _analyze_model_with_direction(
    path: Path, suffix: str, id_cols: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Run analysis pipeline, returning projections AND direction vector."""
    print(f"\n--- Analyzing {suffix} model ---")
    meta, embeddings = load_embeddings(path)
    direction = _compute_direction(meta, embeddings)

    projections = project_onto_direction(embeddings, direction)
    text_col = 'combined_text' if 'combined_text' in meta.columns else 'message_text'
    _print_rankings(meta, projections, text_col)
    _run_probe_validation(direction, suffix)

    proj_df = build_projection_csv(meta, projections, suffix, id_cols)
    return proj_df, direction


def _analyze_messages_cross_level(
    pr_directions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Project messages onto both message-level and player-round directions."""
    results = {}
    for suffix, msg_path in [('small', EMBEDDINGS_SMALL), ('large', EMBEDDINGS_LARGE)]:
        print(f"\n--- Analyzing {suffix} model ---")
        meta, embeddings = load_embeddings(msg_path)
        msg_dir = _compute_direction(meta, embeddings)
        msg_proj = project_onto_direction(embeddings, msg_dir)
        pr_proj = project_onto_direction(embeddings, pr_directions[suffix])
        results[suffix] = (meta, msg_proj, pr_proj)
    return _build_cross_level_output(results)


def _build_cross_level_output(
    results: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Combine small and large projection results into a single output DataFrame."""
    meta = results['small'][0]
    out = meta[ID_COLS + ['player_state']].copy()
    out['proj_msg_dir_small'] = results['small'][1]
    out['proj_pr_dir_small'] = results['small'][2]
    out['proj_msg_dir_large'] = results['large'][1]
    out['proj_pr_dir_large'] = results['large'][2]
    return out


# ===== Data loading =====
def load_embeddings(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load parquet, separating metadata from embedding columns."""
    df = pd.read_parquet(path)
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    meta = df.drop(columns=emb_cols)
    embeddings = df[emb_cols].values
    return meta, embeddings


# ===== Centroid computation =====
def compute_centroids(
    embeddings: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean embedding for cooperative vs non-cooperative groups."""
    coop_mask = labels == STATE_COOPERATIVE
    noncoop_mask = labels == STATE_NONCOOPERATIVE
    if not coop_mask.any():
        raise ValueError(f"No '{STATE_COOPERATIVE}' labels found. Labels: {set(labels)}")
    if not noncoop_mask.any():
        raise ValueError(f"No '{STATE_NONCOOPERATIVE}' labels found. Labels: {set(labels)}")
    coop_centroid = embeddings[coop_mask].mean(axis=0)
    noncoop_centroid = embeddings[noncoop_mask].mean(axis=0)
    return coop_centroid, noncoop_centroid


def compute_difference_vector(
    coop_centroid: np.ndarray, noncoop_centroid: np.ndarray
) -> np.ndarray:
    """Normalized direction: cooperative minus non-cooperative."""
    diff = coop_centroid - noncoop_centroid
    norm = np.linalg.norm(diff)
    if norm == 0:
        raise ValueError(
            "Direction vector has zero norm (centroids identical). "
            "Check that labeling produced distinct groups."
        )
    return diff / norm


# ===== Projection =====
def project_onto_direction(
    embeddings: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Dot product projection. Higher = more cooperative-like."""
    return embeddings @ direction


# ===== Group-round aggregation =====
def compute_group_round_embeddings(
    metadata: pd.DataFrame, embeddings: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    """Mean embedding per group-round. Returns group metadata and embeddings."""
    group_keys = ['session_code', 'segment', 'round', 'group']
    metadata = metadata.copy()
    metadata['_idx'] = np.arange(len(metadata))

    grouped = metadata.groupby(group_keys)['_idx'].apply(list)
    group_meta_rows = []
    group_embs = []

    for keys, indices in grouped.items():
        group_meta_rows.append(dict(zip(group_keys, keys)))
        group_embs.append(embeddings[indices].mean(axis=0))

    group_meta = pd.DataFrame(group_meta_rows)
    return group_meta, np.array(group_embs)


# ===== Ranking =====
def rank_messages(
    metadata: pd.DataFrame, projections: np.ndarray, n: int = 20,
    text_col: str = 'message_text',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top-N most cooperative and non-cooperative texts."""
    df = metadata[[text_col, 'player_state']].copy()
    df['projection'] = projections

    sorted_df = df.sort_values('projection', ascending=False)
    top_coop = sorted_df.head(n).reset_index(drop=True)
    top_noncoop = sorted_df.tail(n).reset_index(drop=True)
    return top_coop, top_noncoop


# ===== Probe phrase validation =====
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


def cosine_similarities(
    embeddings: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Cosine similarity between each embedding and direction vector."""
    norms = np.linalg.norm(embeddings, axis=1)
    # Avoid division by zero
    norms = np.where(norms == 0, 1.0, norms)
    return (embeddings @ direction) / norms


# ===== Output construction =====
def build_projection_csv(
    metadata: pd.DataFrame, projections: np.ndarray, suffix: str,
    id_cols: list[str] | None = None, col_name: str | None = None,
) -> pd.DataFrame:
    """Build output DataFrame with ID columns and projection score."""
    cols = id_cols if id_cols is not None else ID_COLS
    out = metadata[cols + ['player_state']].copy()
    score_col = col_name or f'proj_pr_dir_{suffix}'
    out[score_col] = projections
    return out


# ===== Internal helpers =====
def _compute_direction(
    meta: pd.DataFrame, embeddings: np.ndarray
) -> np.ndarray:
    """Compute normalized direction vector from labeled embeddings."""
    labels = meta['player_state'].values
    coop_c, noncoop_c = compute_centroids(embeddings, labels)
    direction = compute_difference_vector(coop_c, noncoop_c)

    n_coop = (labels == STATE_COOPERATIVE).sum()
    n_noncoop = (labels == STATE_NONCOOPERATIVE).sum()
    print(f"  Cooperative: {n_coop}, Non-cooperative: {n_noncoop}")
    return direction


def _merge_projections(
    proj_small: pd.DataFrame, proj_large: pd.DataFrame,
    id_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Merge small and large projection scores on ID columns."""
    cols = id_cols if id_cols is not None else ID_COLS
    merge_keys = cols + ['player_state']
    large_cols = merge_keys + ['proj_pr_dir_large']
    return proj_small.merge(proj_large[large_cols], on=merge_keys)


def _print_rankings(
    meta: pd.DataFrame, projections: np.ndarray,
    text_col: str = 'message_text',
) -> None:
    """Print top cooperative and non-cooperative messages."""
    top_coop, top_noncoop = rank_messages(meta, projections, n=5, text_col=text_col)
    print("\n  Top cooperative texts:")
    for _, row in top_coop.iterrows():
        print(f"    [{row['projection']:.4f}] {row[text_col][:60]}")
    print("\n  Top non-cooperative texts:")
    for _, row in top_noncoop.iterrows():
        print(f"    [{row['projection']:.4f}] {row[text_col][:60]}")


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
