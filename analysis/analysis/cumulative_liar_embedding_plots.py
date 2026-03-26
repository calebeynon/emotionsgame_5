"""
Visualization of chat message embeddings colored by cumulative liar status.

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
from analyze_cumulative_liar_embeddings import (
    STATE_LIAR,
    STATE_NON_LIAR,
    CLIAR_COL,
)

# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / 'datastore' / 'derived'
EMBEDDINGS_SMALL = DERIVED_DIR / 'embeddings_small.parquet'
EMBEDDINGS_LARGE = DERIVED_DIR / 'embeddings_large.parquet'
LIAR_FILE = DERIVED_DIR / 'behavior_classifications.csv'
PROJECTIONS_FILE = DERIVED_DIR / 'cumulative_liar_embedding_projections.csv'
OUTPUT_DIR = Path(__file__).parent.parent / 'output' / 'plots'

# PLOT STYLING
COLORS = {STATE_LIAR: '#e74c3c', STATE_NON_LIAR: '#2ecc71'}
NAMES = {STATE_LIAR: 'Cumulative Liar', STATE_NON_LIAR: 'Non-Liar'}
FIGSIZE = (8, 6)
JOIN_KEYS = ['session_code', 'segment', 'round', 'group', 'label']


# ===== Main function =====
def main():
    """Generate all cumulative-liar-colored embedding visualizations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    liar_labels = _load_liar_labels()

    for suffix, path in [('small', EMBEDDINGS_SMALL), ('large', EMBEDDINGS_LARGE)]:
        meta, emb = load_embeddings(path)
        labels = _merge_labels(meta, liar_labels)
        _generate_model_plots(emb, labels, suffix)

    _generate_projection_plots()
    print(f"All plots saved to {OUTPUT_DIR}")


def _load_liar_labels() -> pd.DataFrame:
    """Load cumulative-liar labels from behavior classifications."""
    df = pd.read_csv(LIAR_FILE)
    df = df.rename(columns={'round_num': 'round', 'group_id': 'group'})
    df[CLIAR_COL] = df['is_liar_20'].map(
        {True: STATE_LIAR, False: STATE_NON_LIAR}
    )
    return df[JOIN_KEYS + [CLIAR_COL]]


def _merge_labels(meta: pd.DataFrame, liar_df: pd.DataFrame) -> np.ndarray:
    """Join cumulative-liar labels onto embedding metadata."""
    merged = meta.merge(liar_df, on=JOIN_KEYS, how='left')
    return merged[CLIAR_COL].values


def _generate_model_plots(emb: np.ndarray, labels: np.ndarray, suffix: str) -> None:
    """Generate UMAP and t-SNE for one model."""
    valid = np.isin(labels, [STATE_LIAR, STATE_NON_LIAR])
    _plot_reduction(
        UMAP(n_components=2, random_state=42),
        emb[valid], labels[valid],
        f'Embeddings UMAP by Cumulative Liar ({suffix})',
        OUTPUT_DIR / f'cumulative_liar_umap_{suffix}.png',
    )
    _plot_reduction(
        TSNE(n_components=2, random_state=42, perplexity=30),
        emb[valid], labels[valid],
        f'Embeddings t-SNE by Cumulative Liar ({suffix})',
        OUTPUT_DIR / f'cumulative_liar_tsne_{suffix}.png',
    )


# ===== Projection distribution =====
def _generate_projection_plots() -> None:
    """Generate projection distribution plots."""
    proj = pd.read_csv(PROJECTIONS_FILE)
    for suffix in ['small', 'large']:
        col = f'proj_cliar_pr_dir_{suffix}'
        labeled = proj.dropna(subset=[CLIAR_COL, col])
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for state, color in COLORS.items():
            data = labeled.loc[labeled[CLIAR_COL] == state, col]
            ax.hist(data, bins=40, alpha=0.5, color=color,
                    label=NAMES[state], density=True)
        ax.set_xlabel('Projection Score')
        ax.set_ylabel('Density')
        ax.set_title(f'Cumulative-Liar Projection Distribution ({suffix})')
        ax.legend()
        _save_and_close(fig, OUTPUT_DIR / f'cumulative_liar_projection_dist_{suffix}.png')


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
