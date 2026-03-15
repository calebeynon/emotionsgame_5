"""
Visualization of chat message embeddings using UMAP, t-SNE, and distributions.

Generates 2D projections and projection score distributions for both
small and large embedding models, colored by cooperative state.

Author: Claude Code
Date: 2026-03-15
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from umap import UMAP

sys.path.insert(0, str(Path(__file__).parent.parent / "derived"))

from analyze_embeddings import (
    load_embeddings,
    compute_group_round_embeddings,
    STATE_COOPERATIVE,
    STATE_NONCOOPERATIVE,
)

# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / 'datastore' / 'derived'
EMBEDDINGS_SMALL = DERIVED_DIR / 'embeddings_small.parquet'
EMBEDDINGS_LARGE = DERIVED_DIR / 'embeddings_large.parquet'
PROJECTIONS_FILE = DERIVED_DIR / 'embedding_projections.csv'
OUTPUT_DIR = Path(__file__).parent.parent / 'output' / 'plots'

# PLOT STYLING
COOP_COLOR = '#2ecc71'
NONCOOP_COLOR = '#e74c3c'
STATE_COLORS = {STATE_COOPERATIVE: COOP_COLOR, STATE_NONCOOPERATIVE: NONCOOP_COLOR}
FIGSIZE = (8, 6)


# =====
# Main function
# =====
def main():
    """Generate all embedding visualizations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for suffix, path in [('small', EMBEDDINGS_SMALL), ('large', EMBEDDINGS_LARGE)]:
        meta, emb = load_embeddings(path)
        labels = meta['player_state'].values
        _generate_model_plots(emb, labels, meta, suffix)

    _generate_projection_plots()
    print(f"All plots saved to {OUTPUT_DIR}")


# =====
# Per-model plot generation
# =====
def _generate_model_plots(
    emb: np.ndarray, labels: np.ndarray, meta: pd.DataFrame, suffix: str
) -> None:
    """Generate UMAP, t-SNE, and group UMAP for one model."""
    labeled_mask = _get_labeled_mask(labels)

    plot_umap(
        emb[labeled_mask], labels[labeled_mask],
        f'Message Embeddings UMAP ({suffix})',
        OUTPUT_DIR / f'embedding_umap_{suffix}.png',
    )
    plot_tsne(
        emb[labeled_mask], labels[labeled_mask],
        f'Message Embeddings t-SNE ({suffix})',
        OUTPUT_DIR / f'embedding_tsne_{suffix}.png',
    )
    _generate_group_umap(meta, emb, suffix)


def _get_labeled_mask(labels: np.ndarray) -> np.ndarray:
    """Boolean mask for rows with cooperative or non-cooperative labels."""
    return np.isin(labels, [STATE_COOPERATIVE, STATE_NONCOOPERATIVE])


# =====
# UMAP plot
# =====
def plot_umap(
    embeddings: np.ndarray, labels: np.ndarray, title: str, output_path: Path
) -> None:
    """2D UMAP colored by cooperative state."""
    reducer = UMAP(n_components=2, random_state=42)
    coords = reducer.fit_transform(embeddings)
    _save_scatter(coords, labels, title, output_path)


# =====
# t-SNE plot
# =====
def plot_tsne(
    embeddings: np.ndarray, labels: np.ndarray, title: str, output_path: Path
) -> None:
    """2D t-SNE colored by cooperative state."""
    reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = reducer.fit_transform(embeddings)
    _save_scatter(coords, labels, title, output_path)


# =====
# Projection distribution plot
# =====
def plot_projection_distribution(
    projections: pd.DataFrame, suffix: str, output_path: Path
) -> None:
    """Histogram/KDE of projection scores split by state."""
    col = f'projection_score_{suffix}'
    labeled = projections.dropna(subset=['player_state', col])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for state, color in STATE_COLORS.items():
        data = labeled.loc[labeled['player_state'] == state, col]
        ax.hist(data, bins=40, alpha=0.5, color=color, label=state, density=True)

    ax.set_xlabel('Projection Score')
    ax.set_ylabel('Density')
    ax.set_title(f'Projection Score Distribution ({suffix})')
    ax.legend()
    _save_and_close(fig, output_path)


# =====
# Group-round UMAP plot
# =====
def _generate_group_umap(
    meta: pd.DataFrame, emb: np.ndarray, suffix: str
) -> None:
    """UMAP of group-round mean embeddings colored by majority state."""
    group_keys = ['session_code', 'segment', 'round', 'group']
    group_meta, group_emb = compute_group_round_embeddings(meta, emb)
    majority = _assign_majority_state(meta)
    group_meta['majority_state'] = group_meta.merge(
        majority, on=group_keys, how='left',
    )['player_state']
    group_labels = group_meta['majority_state'].values

    if len(group_emb) < 5:
        print(f"  Skipping group UMAP ({suffix}): too few groups")
        return

    plot_group_round_umap(
        group_emb, group_labels,
        OUTPUT_DIR / f'embedding_group_umap_{suffix}.png',
    )


def plot_group_round_umap(
    group_embeddings: np.ndarray, group_labels: np.ndarray, output_path: Path
) -> None:
    """UMAP of group-round mean embeddings."""
    labeled = np.isin(group_labels, [STATE_COOPERATIVE, STATE_NONCOOPERATIVE])
    reducer = UMAP(n_components=2, random_state=42)
    coords = reducer.fit_transform(group_embeddings[labeled])
    _save_scatter(
        coords, group_labels[labeled],
        'Group-Round Embeddings UMAP', output_path,
    )


def _assign_majority_state(meta: pd.DataFrame) -> pd.DataFrame:
    """Assign majority player_state to each group-round as a DataFrame."""
    group_keys = ['session_code', 'segment', 'round', 'group']
    majority = meta.groupby(group_keys)['player_state'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
    ).reset_index()
    return majority


# =====
# Projection distribution wrapper
# =====
def _generate_projection_plots() -> None:
    """Generate projection distribution plots for both models."""
    projections = pd.read_csv(PROJECTIONS_FILE)
    for suffix in ['small', 'large']:
        plot_projection_distribution(
            projections, suffix,
            OUTPUT_DIR / f'embedding_projection_dist_{suffix}.png',
        )


# =====
# Shared plotting helpers
# =====
def _save_scatter(
    coords: np.ndarray, labels: np.ndarray, title: str, output_path: Path
) -> None:
    """Save a 2D scatter plot colored by state labels."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for state, color in STATE_COLORS.items():
        mask = labels == state
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, label=state, alpha=0.5, s=15,
        )

    ax.set_title(title)
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    _save_and_close(fig, output_path)


def _save_and_close(fig: plt.Figure, output_path: Path) -> None:
    """Save figure and close to free memory."""
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# %%
if __name__ == "__main__":
    main()
