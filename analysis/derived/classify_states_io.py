"""
Data loading helpers for state classification module.
Author: Claude Code | Date: 2026-02-20
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_data import Experiment, load_experiment_data

# FILE PATHS
DATA_DIR = Path(__file__).parent.parent / 'datastore'
RAW_DIR = DATA_DIR / 'raw'
PROMISE_FILE = DATA_DIR / 'derived' / 'promise_classifications.csv'


# =====
# Experiment loading
# =====
def load_experiment() -> Experiment:
    """Load experiment data from raw session files."""
    return load_experiment_data(build_file_pairs(), name="State Classification")


def build_file_pairs() -> list:
    """Build list of (data_csv, chat_csv, treatment) tuples from raw directory."""
    file_pairs = []
    for data_file in sorted(RAW_DIR.glob("*_data.csv")):
        treatment = extract_treatment(data_file.name)
        chat_file = data_file.with_name(data_file.name.replace("_data", "_chat"))
        chat_path = str(chat_file) if chat_file.exists() else None
        file_pairs.append((str(data_file), chat_path, treatment))
    return file_pairs


def extract_treatment(filename: str) -> int:
    """Extract treatment number from filename like '01_t1_data.csv'."""
    if '_t1_' in filename:
        return 1
    return 2 if '_t2_' in filename else 0


# =====
# Promise lookup
# =====
def load_promise_lookup(filepath: Path) -> dict:
    """Load promise_classifications.csv into lookup dict.

    Returns:
        dict mapping (session, segment, round, label) -> bool
    """
    if not filepath.exists():
        print(f"Warning: Promise file not found at {filepath}")
        return {}
    df = pd.read_csv(filepath)
    return build_lookup_from_df(df)


def build_lookup_from_df(df: pd.DataFrame) -> dict:
    """Build lookup dict from promise DataFrame."""
    lookup = {}
    for _, row in df.iterrows():
        key = (row['session_code'], row['segment'], row['round'], row['label'])
        lookup[key] = row['promise_count'] > 0
    return lookup


# =====
# DataFrame export helpers
# =====
def build_group_mean_index(group_observations) -> dict:
    """Index group observations by (session, segment, round, group_id) for lookup."""
    return {
        (go.session_code, go.segment, go.round_num, go.group_id): go.mean_contribution
        for go in group_observations
    }


def obs_to_row(obs, state, cell, group_means, classification) -> dict:
    """Convert a single Observation into a flat dict for DataFrame export."""
    key = (obs.session_code, obs.segment, obs.round_num, obs.group_id)
    return {
        'session_code': obs.session_code,
        'treatment': obs.treatment,
        'segment': obs.segment,
        'round_num': obs.round_num,
        'group_id': obs.group_id,
        'label': obs.label,
        'contribution': obs.contribution,
        'group_mean_contribution': group_means.get(key, 0.0),
        'group_state': state.label,
        'player_behavior': cell.behavior_label,
        'made_promise': obs.made_promise,
        'group_threshold': classification.group_threshold,
        'player_threshold': classification.player_threshold,
    }
