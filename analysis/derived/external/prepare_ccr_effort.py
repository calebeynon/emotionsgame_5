"""
Prepare Chen, Chen & Riyanto (2021) effort data for external validation.

Loads the CCR Stata file and produces two parquet outputs:
  1. Subject-period panel (34,800 rows) for panel regressions
  2. Group-level cross-section (116 rows) for cross-sectional analysis

Author: Claude Code
Date: 2026-03-26
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# FILE PATHS
DATA_FILE = Path(
    "analysis/datastore/raw/external_datasets"
    "/chen_chen_riyanto_2021/4 - Data Analysis/Statistics/data.dta"
)
OUTPUT_DIR = Path("analysis/datastore/derived/external")
PANEL_OUTPUT = OUTPUT_DIR / "ccr_effort_panel.parquet"
GROUP_OUTPUT = OUTPUT_DIR / "ccr_effort_group.parquet"

# Column selections
CORE_COLS = [
    "session", "run", "subject", "red", "period",
    "effort", "matcheffort", "ingroup", "commonknow",
]
DEMOGRAPHIC_COLS = [
    "age", "female",
    "raceCaucasian", "raceBlack", "raceHispanic",
    "raceNatAmerican", "raceMulti", "raceOther",
    "continentNAmerica", "continentEurope", "continentAfrica",
    "continentAustralia", "continentSAmerica",
]
CHAT_STAT_COLS = ["count_lines", "count_words", "count_characters"]
CHAT_CODED_COLS = [
    "analysis", "question", "answer", "agreement",
    "disagreement", "experiment", "group", "excitement", "irrelevant",
]


# =====
# Main function
# =====
def main():
    """Load CCR data, build panel and cross-section, save to parquet."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raw = _load_data()
    panel = _build_panel(raw)
    _save_panel(panel)
    group = _build_group_cross_section(panel)
    _save_group(group)
    _print_summary(panel, group)


# =====
# Data loading
# =====
def _load_data():
    """Load raw Stata file and validate expected shape."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"CCR data file not found: {DATA_FILE}. "
            "Check that the datastore symlink is set up."
        )
    df = pd.read_stata(DATA_FILE)
    logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), DATA_FILE)
    if len(df) != 34800:
        raise ValueError(f"Expected 34800 rows, got {len(df)}")
    return df


# =====
# Panel construction
# =====
def _build_panel(raw):
    """Select columns and add merge key for the subject-period panel."""
    keep_cols = CORE_COLS + DEMOGRAPHIC_COLS + CHAT_STAT_COLS + CHAT_CODED_COLS
    panel = raw[keep_cols].copy()
    # Cast integer-valued columns to int for cleaner output
    int_cols = ["session", "run", "subject", "red", "period"]
    for col in int_cols:
        panel[col] = panel[col].astype(int)
    # Merge key: (session, red) identifies the color team
    panel["group_key"] = (
        panel["session"].astype(str) + "_" + panel["red"].astype(str)
    )
    return panel


def _save_panel(panel):
    """Write panel parquet to disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_OUTPUT, index=False)
    logger.info("Panel saved: %d rows -> %s", len(panel), PANEL_OUTPUT)


# =====
# Group-level cross-section
# =====
def _build_group_cross_section(panel):
    """Aggregate panel to one row per (session, red) color team."""
    grp = panel.groupby(["session", "red", "group_key"])
    effort_agg = _aggregate_effort(panel)
    period1_agg = _aggregate_period1(panel)
    treatment = _extract_treatment_indicators(grp)
    chat_stats = _aggregate_chat_stats(panel)
    demographics = _aggregate_demographics(panel)
    group = (
        effort_agg
        .merge(period1_agg, on="group_key")
        .merge(treatment, on="group_key")
        .merge(chat_stats, on="group_key")
        .merge(demographics, on="group_key")
    )
    if len(group) != 116:
        raise ValueError(f"Expected 116 groups, got {len(group)}")
    return group


def _aggregate_effort(panel):
    """Compute mean and min effort across all periods per group."""
    agg = panel.groupby("group_key")["effort"].agg(
        mean_effort="mean", min_effort="min"
    ).reset_index()
    return agg


def _aggregate_period1(panel):
    """Compute mean and min effort for period 1 only."""
    p1 = panel[panel["period"] == 1]
    agg = p1.groupby("group_key")["effort"].agg(
        mean_effort_period1="mean", min_effort_period1="min"
    ).reset_index()
    return agg


def _extract_treatment_indicators(grp):
    """Extract constant treatment indicators per group."""
    treatment = grp[["ingroup", "commonknow", "run"]].first().reset_index()
    return treatment[["group_key", "session", "red", "ingroup", "commonknow", "run"]]


def _aggregate_chat_stats(panel):
    """Average chat statistics across team members (constant per subject)."""
    # Chat stats are constant across periods, so take first period only
    p1 = panel[panel["period"] == 1]
    all_chat_cols = CHAT_STAT_COLS + CHAT_CODED_COLS
    agg = p1.groupby("group_key")[all_chat_cols].mean().reset_index()
    agg = agg.rename(columns={c: f"mean_{c}" for c in all_chat_cols})
    return agg


def _aggregate_demographics(panel):
    """Average demographics across team members."""
    p1 = panel[panel["period"] == 1]
    agg = p1.groupby("group_key")[DEMOGRAPHIC_COLS].mean().reset_index()
    agg = agg.rename(columns={c: f"mean_{c}" for c in DEMOGRAPHIC_COLS})
    return agg


def _save_group(group):
    """Write group cross-section parquet to disk."""
    group.to_parquet(GROUP_OUTPUT, index=False)
    logger.info("Group cross-section saved: %d rows -> %s", len(group), GROUP_OUTPUT)


# =====
# Summary reporting
# =====
def _print_summary(panel, group):
    """Print verification statistics."""
    _print_panel_summary(panel)
    _print_group_summary(group)


def _print_panel_summary(panel):
    """Print panel-level statistics."""
    logger.info("--- Panel Summary ---")
    logger.info("Shape: %s", panel.shape)
    logger.info(
        "Effort: mean=%.1f, sd=%.1f, min=%d, max=%d",
        panel["effort"].mean(), panel["effort"].std(),
        int(panel["effort"].min()), int(panel["effort"].max()),
    )
    logger.info("Unique groups: %d", panel["group_key"].nunique())


def _print_group_summary(group):
    """Print group cross-section statistics."""
    logger.info("--- Group Cross-Section ---")
    logger.info("Shape: %s", group.shape)
    logger.info(
        "Mean effort: %.1f (sd=%.1f)",
        group["mean_effort"].mean(), group["mean_effort"].std(),
    )
    logger.info(
        "Period-1 mean effort: %.1f (sd=%.1f)",
        group["mean_effort_period1"].mean(),
        group["mean_effort_period1"].std(),
    )
    logger.info(
        "Treatment split: ingroup=%d, outgroup=%d",
        int(group["ingroup"].sum()), int((1 - group["ingroup"]).sum()),
    )


# %%
if __name__ == "__main__":
    main()
