"""
Visualization of chat message embeddings colored by promise-making behavior.

Generates UMAP, t-SNE, projection distributions, and group-level UMAP
for both small and large embedding models.

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
)

# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / 'datastore' / 'derived'
EMBEDDINGS_SMALL = DERIVED_DIR / 'embeddings_small.parquet'
EMBEDDINGS_LARGE = DERIVED_DIR / 'embeddings_large.parquet'
STATE_FILE = DERIVED_DIR / 'player_state_classification.csv'
PROJECTIONS_FILE = DERIVED_DIR / 'promise_embedding_projections.csv'
OUTPUT_DIR = Path(__file__).parent.parent / 'output' / 'plots'

# PLOT STYLING
PROMISE_COLOR = '#3498db'
NO_PROMISE_COLOR = '#e67e22'
LABEL_COLORS = {True: PROMISE_COLOR, False: NO_PROMISE_COLOR}
LABEL_NAMES = {True: 'Promise', False: 'No Promise'}
PROMISE_LABEL_COL = 'promise_label'
PROJ_LABEL_COLORS = {'promise': PROMISE_COLOR, 'no_promise': NO_PROMISE_COLOR}
PROJ_LABEL_NAMES = {'promise': 'Promise', 'no_promise': 'No Promise'}
FIGSIZE = (8, 6)


# =====
# Main function
# =====
def main():
    """Generate all promise-colored embedding visualizations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    promise_labels = _load_promise_labels()

    for suffix, path in [('small', EMBEDDINGS_SMALL), ('large', EMBEDDINGS_LARGE)]:
        meta, emb = load_embeddings(path)
        labels = _merge_promise_labels(meta, promise_labels)
        _generate_model_plots(emb, labels, meta, suffix)

    _generate_projection_plots()
    print(f"All plots saved to {OUTPUT_DIR}")


# =====
# Promise label loading
# =====
def _load_promise_labels() -> pd.DataFrame:
    """Load made_promise from player state classification."""
    df = pd.read_csv(STATE_FILE)
    df = df.rename(columns={'round_num': 'round', 'group_id': 'group'})
    join_cols = ['session_code', 'segment', 'round', 'group', 'label']
    return df[join_cols + ['made_promise']].copy()


def _merge_promise_labels(
    meta: pd.DataFrame, promise_df: pd.DataFrame
) -> np.ndarray:
    """Join made_promise onto embedding metadata, return boolean array."""
    join_cols = ['session_code', 'segment', 'round', 'group', 'label']
    merged = meta.merge(promise_df, on=join_cols, how='left')
    return merged['made_promise'].values


# =====
# Per-model plot generation
# =====
def _generate_model_plots(
    emb: np.ndarray, labels: np.ndarray, meta: pd.DataFrame, suffix: str
) -> None:
    """Generate UMAP, t-SNE, and group UMAP for one model."""
    valid = ~pd.isna(labels)

    _plot_scatter_reduction(
        UMAP(n_components=2, random_state=42),
        emb[valid], labels[valid],
        f'Message Embeddings UMAP by Promise ({suffix})',
        OUTPUT_DIR / f'promise_umap_{suffix}.png',
    )
    _plot_scatter_reduction(
        TSNE(n_components=2, random_state=42, perplexity=30),
        emb[valid], labels[valid],
        f'Message Embeddings t-SNE by Promise ({suffix})',
        OUTPUT_DIR / f'promise_tsne_{suffix}.png',
    )
    _generate_group_umap(meta, emb, labels, suffix)


# =====
# Dimensionality reduction scatter
# =====
def _plot_scatter_reduction(
    reducer, embeddings: np.ndarray, labels: np.ndarray,
    title: str, output_path: Path,
) -> None:
    """Fit reducer and save scatter plot colored by promise label."""
    coords = reducer.fit_transform(embeddings)
    _save_scatter(coords, labels, title, output_path)


# =====
# Group-round UMAP
# =====
def _generate_group_umap(
    meta: pd.DataFrame, emb: np.ndarray, labels: np.ndarray, suffix: str
) -> None:
    """UMAP of group-round mean embeddings colored by majority promise."""
    group_meta, group_emb = compute_group_round_embeddings(meta, emb)
    group_labels = _compute_group_labels(meta, labels, group_meta)

    valid = ~pd.isna(group_labels)
    if valid.sum() < 5:
        print(f"  Skipping group UMAP ({suffix}): too few groups")
        return

    coords = UMAP(n_components=2, random_state=42).fit_transform(group_emb[valid])
    _save_scatter(
        coords, group_labels[valid],
        f'Group-Round Embeddings UMAP by Promise ({suffix})',
        OUTPUT_DIR / f'promise_group_umap_{suffix}.png',
    )


def _compute_group_labels(
    meta: pd.DataFrame, labels: np.ndarray, group_meta: pd.DataFrame
) -> np.ndarray:
    """Merge majority promise labels onto group metadata."""
    group_keys = ['session_code', 'segment', 'round', 'group']
    majority = _assign_majority_promise(meta, labels)
    merged = group_meta.merge(majority, on=group_keys, how='left')
    return merged['made_promise'].values


def _assign_majority_promise(
    meta: pd.DataFrame, labels: np.ndarray
) -> pd.DataFrame:
    """Assign majority made_promise to each group-round."""
    group_keys = ['session_code', 'segment', 'round', 'group']
    df = meta[group_keys].copy()
    df['made_promise'] = labels
    majority = df.groupby(group_keys)['made_promise'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
    ).reset_index()
    return majority


# =====
# Projection distribution plots
# =====
def _generate_projection_plots() -> None:
    """Generate projection distribution plots if projections exist."""
    if not PROJECTIONS_FILE.exists():
        print(f"  Skipping projection plots: {PROJECTIONS_FILE.name} not found")
        return

    projections = pd.read_csv(PROJECTIONS_FILE)
    for suffix in ['small', 'large']:
        _plot_projection_distribution(
            projections, suffix,
            OUTPUT_DIR / f'promise_projection_dist_{suffix}.png',
        )


def _plot_projection_distribution(
    projections: pd.DataFrame, suffix: str, output_path: Path
) -> None:
    """Histogram of projection scores split by promise status."""
    col = f'proj_promise_msg_dir_{suffix}'
    if col not in projections.columns:
        print(f"  Skipping {output_path.name}: column {col} not found")
        return

    labeled = projections.dropna(subset=[PROMISE_LABEL_COL, col])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    _draw_promise_histogram(ax, labeled, col)
    ax.set_title(f'Promise Projection Score Distribution ({suffix})')
    _save_and_close(fig, output_path)


def _draw_promise_histogram(
    ax: plt.Axes, labeled: pd.DataFrame, col: str
) -> None:
    """Draw overlaid histograms for promise vs no-promise groups."""
    for value, color in PROJ_LABEL_COLORS.items():
        data = labeled.loc[labeled[PROMISE_LABEL_COL] == value, col]
        ax.hist(
            data, bins=40, alpha=0.5, color=color,
            label=PROJ_LABEL_NAMES[value], density=True,
        )
    ax.set_xlabel('Projection Score')
    ax.set_ylabel('Density')
    ax.legend()


# =====
# Shared plotting helpers
# =====
def _save_scatter(
    coords: np.ndarray, labels: np.ndarray, title: str, output_path: Path
) -> None:
    """Save a 2D scatter plot colored by promise labels."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for value, color in LABEL_COLORS.items():
        mask = labels == value
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, label=LABEL_NAMES[value], alpha=0.5, s=15,
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
