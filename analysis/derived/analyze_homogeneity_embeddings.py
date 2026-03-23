"""
Analyze chat embeddings along the homogeneous vs heterogeneous contribution axis.

Computes homogeneity labels (contribution range <= 1 ECU per group-round),
projects embeddings onto direction vector, and outputs projection scores.

Author: Claude Code
Date: 2026-03-20
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
SENTIMENT_FILE = DERIVED_DIR / 'sentiment_scores.csv'
PROJECTIONS_OUTPUT = DERIVED_DIR / 'homogeneity_embedding_projections.csv'

# ANALYSIS CONSTANTS
HOMOG_COL = 'homogeneity_label'
STATE_HOMOGENEOUS = 'homogeneous'
STATE_HETEROGENEOUS = 'heterogeneous'

PROBE_PHRASES = [
    "let's all put in the same amount",
    "I'll match whatever you do",
    "I'm putting in zero",
    "everyone do 25",
    "we should all contribute equally",
    "I'll do what the group does",
    "I'm going my own way",
    "some of us are free riding",
    "we're all on the same page",
    "I don't care what others do",
    "let's coordinate our contributions",
]

GROUP_KEYS = ['session_code', 'segment', 'round', 'group']
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
    homog_df = compute_homogeneity_labels()
    pr_dir = _run_player_round_analysis(homog_df)
    _run_message_analysis(pr_dir, homog_df)


def _run_player_round_analysis(homog_df: pd.DataFrame) -> dict:
    """Compute player-round projections and save. Returns direction vectors."""
    print("=== Player-round level analysis ===")
    pr_dir = {}
    pr_small, pr_dir['small'] = _analyze_model_with_direction(
        EMBEDDINGS_PR_SMALL, 'small', PR_ID_COLS, homog_df,
    )
    pr_large, pr_dir['large'] = _analyze_model_with_direction(
        EMBEDDINGS_PR_LARGE, 'large', PR_ID_COLS, homog_df,
    )
    print(f"\nPlayer-round analysis complete: {len(pr_small)} rows")
    return pr_dir


def _run_message_analysis(pr_dir: dict, homog_df: pd.DataFrame) -> None:
    """Compute message-level cross-level projections and save CSV."""
    print("\n=== Message-level analysis ===")
    combined = _analyze_messages_cross_level(pr_dir, homog_df)
    combined.to_csv(PROJECTIONS_OUTPUT, index=False)
    print(f"\nSaved {len(combined)} rows to {PROJECTIONS_OUTPUT.name}")


# =====
# Homogeneity label computation
# =====
def compute_homogeneity_labels(
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Label each group-round as homogeneous (range <= 1) or heterogeneous."""
    if df is None:
        df = pd.read_csv(SENTIMENT_FILE)
    group_range = _compute_contribution_range(df)
    group_range[HOMOG_COL] = _assign_labels(group_range['contribution_range'])
    _print_label_counts(group_range)
    return group_range[GROUP_KEYS + [HOMOG_COL]]


def _compute_contribution_range(df: pd.DataFrame) -> pd.DataFrame:
    """Compute max - min contribution per group-round."""
    agg = df.groupby(GROUP_KEYS)['contribution'].agg(['max', 'min'])
    agg['contribution_range'] = agg['max'] - agg['min']
    return agg.reset_index()


def _assign_labels(contribution_range: pd.Series) -> pd.Series:
    """Map contribution range to homogeneity labels."""
    return contribution_range.map(
        lambda r: None if pd.isna(r) else (STATE_HOMOGENEOUS if r <= 1 else STATE_HETEROGENEOUS)
    )


def _print_label_counts(group_range: pd.DataFrame) -> None:
    """Print distribution of homogeneity labels."""
    counts = group_range[HOMOG_COL].value_counts()
    print(f"Homogeneity labels: {dict(counts)}")


def merge_homogeneity_labels(
    meta: pd.DataFrame, homog_df: pd.DataFrame,
) -> pd.DataFrame:
    """LEFT JOIN homogeneity labels onto embedding metadata."""
    return meta.merge(homog_df, on=GROUP_KEYS, how='left')


def compute_homogeneity_centroids(
    embeddings: np.ndarray, labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean embedding for homogeneous vs heterogeneous groups."""
    homog_mask = labels == STATE_HOMOGENEOUS
    heterog_mask = labels == STATE_HETEROGENEOUS
    if not homog_mask.any():
        raise ValueError(f"No '{STATE_HOMOGENEOUS}' labels found. Labels: {set(labels)}")
    if not heterog_mask.any():
        raise ValueError(f"No '{STATE_HETEROGENEOUS}' labels found. Labels: {set(labels)}")
    homog_centroid = embeddings[homog_mask].mean(axis=0)
    heterog_centroid = embeddings[heterog_mask].mean(axis=0)
    return homog_centroid, heterog_centroid


# =====
# Direction computation
# =====
def _compute_homogeneity_direction(
    meta: pd.DataFrame, embeddings: np.ndarray,
) -> np.ndarray:
    """Compute normalized homogeneity direction vector."""
    labels = meta[HOMOG_COL].values
    homog_c, heterog_c = compute_homogeneity_centroids(embeddings, labels)
    direction = compute_difference_vector(homog_c, heterog_c)

    n_homog = (labels == STATE_HOMOGENEOUS).sum()
    n_heterog = (labels == STATE_HETEROGENEOUS).sum()
    print(f"  Homogeneous: {n_homog}, Heterogeneous: {n_heterog}")
    return direction


# =====
# Model-level analysis
# =====
def _analyze_model_with_direction(
    path: Path, suffix: str, id_cols: list[str],
    homog_df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Run analysis pipeline, returning projections AND direction vector."""
    print(f"\n--- Analyzing {suffix} model ---")
    meta, embeddings = load_embeddings(path)
    meta = merge_homogeneity_labels(meta, homog_df)

    direction = _compute_homogeneity_direction(meta, embeddings)
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
    homog_df: pd.DataFrame,
) -> pd.DataFrame:
    """Project messages onto both message-level and player-round directions."""
    results = {}
    for suffix, msg_path in [('small', EMBEDDINGS_SMALL), ('large', EMBEDDINGS_LARGE)]:
        print(f"\n--- Analyzing {suffix} model ---")
        meta, embeddings = load_embeddings(msg_path)
        meta = merge_homogeneity_labels(meta, homog_df)

        msg_dir = _compute_homogeneity_direction(meta, embeddings)
        msg_proj = project_onto_direction(embeddings, msg_dir)
        pr_proj = project_onto_direction(embeddings, pr_directions[suffix])
        results[suffix] = (meta, msg_proj, pr_proj)

    return _build_cross_level_output(results)


def _build_cross_level_output(results: dict) -> pd.DataFrame:
    """Build combined output with both projection types."""
    meta = results['small'][0]
    out = meta[ID_COLS + ['player_state', HOMOG_COL]].copy()
    out['proj_homog_msg_dir_small'] = results['small'][1]
    out['proj_homog_pr_dir_small'] = results['small'][2]
    out['proj_homog_msg_dir_large'] = results['large'][1]
    out['proj_homog_pr_dir_large'] = results['large'][2]
    return out


# =====
# Output construction
# =====
def _build_output(
    meta: pd.DataFrame, projections: np.ndarray,
    suffix: str, id_cols: list[str],
) -> pd.DataFrame:
    """Build output DataFrame with ID columns and projection score."""
    out = meta[id_cols + ['player_state', HOMOG_COL]].copy()
    out[f'proj_homog_pr_dir_{suffix}'] = projections
    return out


# =====
# Ranking
# =====
def _print_rankings(
    meta: pd.DataFrame, projections: np.ndarray,
    text_col: str = 'message_text',
) -> None:
    """Print top homogeneous and heterogeneous messages by projection."""
    df = meta[[text_col, HOMOG_COL]].copy()
    df['projection'] = projections
    sorted_df = df.sort_values('projection', ascending=False)

    print("\n  Top homogeneous-direction texts:")
    for _, row in sorted_df.head(5).iterrows():
        print(f"    [{row['projection']:.4f}] {row[text_col][:60]}")

    print("\n  Top heterogeneous-direction texts:")
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
