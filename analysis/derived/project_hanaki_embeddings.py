"""
Project Hanaki & Ozkes embeddings onto cached direction vectors.

Loads embeddings and 5 direction vectors, computes dot-product projections
for each behavioral axis, and outputs a CSV with projections and metadata.

Author: Claude Code
Date: 2026-03-26
"""

from pathlib import Path

import numpy as np
import pandas as pd

# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / 'datastore' / 'derived'
EMBEDDINGS_FILE = DERIVED_DIR / 'hanaki_ozkes_embeddings.parquet'
DIRECTION_DIR = DERIVED_DIR / 'direction_vectors'
OUTPUT_FILE = DERIVED_DIR / 'hanaki_ozkes_projections.csv'

# DIRECTION VECTOR NAMES (order matches output columns)
DIRECTION_NAMES = [
    'cooperative', 'promise', 'homogeneity', 'round_liar', 'cumulative_liar',
]

# OUTPUT COLUMN ORDER
META_COLS = [
    'session_file', 'period', 'player_id', 'group', 'chat_text',
    'Inv', 'OtherInv', 'PairAveCho', 'Profit', 'Chat', 'Comp',
]


# =====
# Main function
# =====
def main():
    """Project embeddings onto direction vectors and save CSV."""
    meta, embeddings = load_embeddings(EMBEDDINGS_FILE)
    directions = load_direction_vectors(DIRECTION_DIR)
    projections = compute_all_projections(embeddings, directions)
    output = build_output(meta, projections)
    output.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(output)} rows to {OUTPUT_FILE.name}")


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


def load_direction_vectors(directory: Path) -> dict[str, np.ndarray]:
    """Load all .npy direction vectors from directory."""
    directions = {}
    for name in DIRECTION_NAMES:
        path = directory / f'{name}.npy'
        directions[name] = np.load(path)
        print(f"  Loaded {name}.npy (shape={directions[name].shape})")
    return directions


# =====
# Projection computation
# =====
def compute_all_projections(
    embeddings: np.ndarray, directions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compute dot-product projections onto each direction vector."""
    proj_data = {}
    for name in DIRECTION_NAMES:
        proj_data[f'proj_{name}'] = embeddings @ directions[name]
    return pd.DataFrame(proj_data)


# =====
# Output construction
# =====
def build_output(meta: pd.DataFrame, projections: pd.DataFrame) -> pd.DataFrame:
    """Combine metadata with PairAveCho and projection columns."""
    meta = meta.copy()
    meta['PairAveCho'] = (meta['Inv'] + meta['OtherInv']) / 2
    proj_cols = [f'proj_{name}' for name in DIRECTION_NAMES]
    output = pd.concat([meta[META_COLS], projections[proj_cols]], axis=1)
    return output


# %%
if __name__ == "__main__":
    main()
