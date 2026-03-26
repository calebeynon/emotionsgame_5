"""
Visualization of chat message embeddings colored by contribution homogeneity.

Author: Claude Code
Date: 2026-03-26
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from umap import UMAP

sys.path.insert(0, str(Path(__file__).parent.parent / "derived"))

from analyze_embeddings import load_embeddings
from analyze_homogeneity_embeddings import (
    compute_homogeneity_labels,
    STATE_HOMOGENEOUS,
    STATE_HETEROGENEOUS,
    HOMOG_COL,
)

# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / 'datastore' / 'derived'
EMBEDDINGS_SMALL = DERIVED_DIR / 'embeddings_small.parquet'
EMBEDDINGS_LARGE = DERIVED_DIR / 'embeddings_large.parquet'
PROJECTIONS_FILE = DERIVED_DIR / 'homogeneity_embedding_projections.csv'
OUTPUT_DIR = Path(__file__).parent.parent / 'output' / 'plots'

# PLOT STYLING
COLORS = {STATE_HOMOGENEOUS: '#2ecc71', STATE_HETEROGENEOUS: '#e74c3c'}
NAMES = {STATE_HOMOGENEOUS: 'Homogeneous', STATE_HETEROGENEOUS: 'Heterogeneous'}
FIGSIZE = (8, 6)
JOIN_KEYS = ['session_code', 'segment', 'round', 'group']


# ===== Main function =====
def main():
    """Generate all homogeneity-colored embedding visualizations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    homog_labels = compute_homogeneity_labels()

    for suffix, path in [('small', EMBEDDINGS_SMALL), ('large', EMBEDDINGS_LARGE)]:
        meta, emb = load_embeddings(path)
        labels = _merge_labels(meta, homog_labels)
        _generate_model_plots(emb, labels, suffix)

    _generate_projection_plots()
    print(f"All plots saved to {OUTPUT_DIR}")


def _merge_labels(meta: pd.DataFrame, homog_df: pd.DataFrame) -> np.ndarray:
    """Join homogeneity labels onto embedding metadata."""
    merged = meta.merge(homog_df[JOIN_KEYS + [HOMOG_COL]], on=JOIN_KEYS, how='left')
    return merged[HOMOG_COL].values


def _generate_model_plots(emb: np.ndarray, labels: np.ndarray, suffix: str) -> None:
    """Generate UMAP and t-SNE for one model."""
    valid = np.isin(labels, [STATE_HOMOGENEOUS, STATE_HETEROGENEOUS])
    _plot_reduction(
        UMAP(n_components=2, random_state=42),
        emb[valid], labels[valid],
        f'Embeddings UMAP by Homogeneity ({suffix})',
        OUTPUT_DIR / f'homogeneity_umap_{suffix}.png',
    )
    _plot_reduction(
        TSNE(n_components=2, random_state=42, perplexity=30),
        emb[valid], labels[valid],
        f'Embeddings t-SNE by Homogeneity ({suffix})',
        OUTPUT_DIR / f'homogeneity_tsne_{suffix}.png',
    )


# ===== Projection distribution =====
def _generate_projection_plots() -> None:
    """Generate projection distribution plots."""
    proj = pd.read_csv(PROJECTIONS_FILE)
    for suffix in ['small', 'large']:
        col = f'proj_homog_pr_dir_{suffix}'
        labeled = proj.dropna(subset=[HOMOG_COL, col])
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for state, color in COLORS.items():
            data = labeled.loc[labeled[HOMOG_COL] == state, col]
            ax.hist(data, bins=40, alpha=0.5, color=color,
                    label=NAMES[state], density=True)
        ax.set_xlabel('Projection Score')
        ax.set_ylabel('Density')
        ax.set_title(f'Homogeneity Projection Distribution ({suffix})')
        ax.legend()
        _save_and_close(fig, OUTPUT_DIR / f'homogeneity_projection_dist_{suffix}.png')


# ===== Dimensionality reduction =====
def _plot_reduction(reducer, emb, labels, title, output_path):
    """Fit reducer and save scatter plot."""
    coords = reducer.fit_transform(emb)
    _save_scatter(coords, labels, title, output_path)


# ===== Shared plotting helpers =====
def _save_scatter(coords, labels, title, output_path):
    """Save a 2D scatter plot colored by labels."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for state, color in COLORS.items():
        mask = labels == state
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, label=NAMES[state], alpha=0.5, s=15)
    ax.set_title(title)
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    _save_and_close(fig, output_path)


def _save_and_close(fig, output_path):
    """Save figure and close."""
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# %%
if __name__ == "__main__":
    main()
