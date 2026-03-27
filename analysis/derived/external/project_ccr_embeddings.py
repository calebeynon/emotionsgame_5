"""
Project CCR chat embeddings onto cached direction vectors.

Loads group-level embeddings and 5 direction vectors, computes dot-product
projections for each behavioral axis, merges with effort panel data, and
outputs panel and group-level parquet files ready for regressions.

Author: Claude Code
Date: 2026-03-26
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# FILE PATHS
DERIVED_DIR = Path("analysis/datastore/derived")
EMBEDDINGS_FILE = DERIVED_DIR / "external" / "ccr_embeddings_small.parquet"
PANEL_FILE = DERIVED_DIR / "external" / "ccr_effort_panel.parquet"
GROUP_FILE = DERIVED_DIR / "external" / "ccr_effort_group.parquet"
DIRECTION_DIR = DERIVED_DIR / "direction_vectors"
OUTPUT_PANEL = DERIVED_DIR / "external" / "ccr_projections_panel.parquet"
OUTPUT_GROUP = DERIVED_DIR / "external" / "ccr_projections_group.parquet"
# CSV exports for R (no arrow package available)
OUTPUT_PANEL_CSV = DERIVED_DIR / "external" / "ccr_projections_panel.csv"
OUTPUT_GROUP_CSV = DERIVED_DIR / "external" / "ccr_projections_group.csv"

# DIRECTION VECTOR NAMES
DIRECTION_NAMES = [
    "cooperative", "promise", "homogeneity", "round_liar", "cumulative_liar",
]


# =====
# Main function
# =====
def main():
    """Project embeddings, merge with effort data, save panel and group files."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    projections = _compute_projections()
    panel = _merge_panel(projections)
    group = _merge_group(projections)
    _save_outputs(panel, group)
    _print_summary(panel, group)


# =====
# Projection computation
# =====
def _compute_projections():
    """Load embeddings and direction vectors, compute dot products."""
    emb_df = pd.read_parquet(EMBEDDINGS_FILE)
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    embeddings = emb_df[emb_cols].values
    group_keys = emb_df["group_key"].values
    directions = _load_direction_vectors()
    proj_data = {"group_key": group_keys}
    for name in DIRECTION_NAMES:
        proj_data[f"proj_{name}"] = embeddings @ directions[name]
    projections = pd.DataFrame(proj_data)
    logger.info("Computed projections for %d groups", len(projections))
    return projections


def _load_direction_vectors():
    """Load all .npy direction vectors."""
    directions = {}
    for name in DIRECTION_NAMES:
        path = DIRECTION_DIR / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Direction vector not found: {path}")
        directions[name] = np.load(path)
        logger.info("  Loaded %s.npy (dim=%d)", name, len(directions[name]))
    return directions


# =====
# Panel merge (subject-period level)
# =====
def _merge_panel(projections):
    """Merge projections onto the subject-period panel via group_key."""
    panel = pd.read_parquet(PANEL_FILE)
    merged = panel.merge(projections, on="group_key", how="left")
    _report_merge_stats("Panel", panel, merged)
    return merged


# =====
# Group-level merge
# =====
def _merge_group(projections):
    """Merge projections onto the group cross-section via group_key."""
    group = pd.read_parquet(GROUP_FILE)
    merged = group.merge(projections, on="group_key", how="left")
    _report_merge_stats("Group", group, merged)
    return merged


def _report_merge_stats(label, original, merged):
    """Log merge diagnostics."""
    proj_cols = [f"proj_{n}" for n in DIRECTION_NAMES]
    n_missing = merged[proj_cols[0]].isna().sum()
    logger.info(
        "%s merge: %d rows, %d with projections, %d missing",
        label, len(merged), len(merged) - n_missing, n_missing,
    )


# =====
# Output
# =====
def _save_outputs(panel, group):
    """Write parquet and CSV output files."""
    panel.to_parquet(OUTPUT_PANEL, index=False)
    panel.to_csv(OUTPUT_PANEL_CSV, index=False)
    logger.info("Saved panel: %s (.parquet + .csv)", OUTPUT_PANEL.stem)
    group.to_parquet(OUTPUT_GROUP, index=False)
    group.to_csv(OUTPUT_GROUP_CSV, index=False)
    logger.info("Saved group: %s (.parquet + .csv)", OUTPUT_GROUP.stem)


def _print_summary(panel, group):
    """Print projection score statistics."""
    proj_cols = [f"proj_{n}" for n in DIRECTION_NAMES]
    logger.info("--- Projection Score Statistics (group-level) ---")
    for col in proj_cols:
        vals = group[col].dropna()
        logger.info(
            "  %s: mean=%.3f, sd=%.3f, range=[%.3f, %.3f]",
            col, vals.mean(), vals.std(), vals.min(), vals.max(),
        )


# %%
if __name__ == "__main__":
    main()
