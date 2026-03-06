"""
Purpose: Shared utilities for summary statistics scripts. Provides constants,
         data loading functions, and LaTeX table output for the summary
         statistics analysis pipeline.
Author: Caleb Eynon
Date: 2026-03-01
"""

import re
from pathlib import Path

import pandas as pd

# FILE PATHS
OUTPUT_DIR = Path('analysis/output/summary_statistics')
DERIVED_DIR = Path('analysis/datastore/derived')
RAW_DIR = Path('analysis/datastore/raw')
SESSIONS_DIR = Path('analysis/datastore/sessions')

# EXPERIMENT CONSTANTS
POINTS_TO_DOLLARS = 0.10
PARTICIPATION_FEE = 7.50
SUPERGAME_ROUNDS = {1: 3, 2: 4, 3: 3, 4: 7, 5: 5}
# Session 03 raw data uses z8dowljr, but derived CSVs use irrzlgk2
SESSION_CODE_REMAP = {'z8dowljr': 'irrzlgk2'}

SESSION_CODE_TO_TREATMENT = {
    'sa7mprty': 1,
    'irrzlgk2': 2,
    '6uv359rf': 2,
    'umbzdj98': 1,
    'j3ki5tli': 2,
    'r5dj4yfl': 1,
    'sylq2syi': 2,
    'iiu3xixz': 1,
    '6ucza025': 2,
    '6sdkxl2q': 1,
}

# Session with invalid rows that must be filtered
_INVALID_SESSION_FILE = '03_t2_data.csv'
_VALID_SESSION_CODE_03 = 'z8dowljr'


# =====
# Output helpers
# =====

def ensure_output_dir():
    """Create the output directory if it does not exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_tex_table(df, filename, column_formats=None):
    """Write a DataFrame as a booktabs LaTeX table to OUTPUT_DIR.

    Produces a minimal tabular environment with toprule/midrule/bottomrule,
    matching the format of existing .tex tables in the project.
    """
    ensure_output_dir()
    if column_formats is None:
        column_formats = 'l' * len(df.columns)
    lines = _build_tex_lines(df, column_formats)
    output_path = OUTPUT_DIR / filename
    output_path.write_text('\n'.join(lines) + '\n')


def _build_tex_lines(df, column_formats):
    """Build list of LaTeX lines for a booktabs table."""
    lines = [
        f'\\begin{{tabular}}{{{column_formats}}}',
        '  \\toprule',
    ]
    header = ' & '.join(str(c) for c in df.columns) + ' \\\\ '
    lines.append('  ' + header)
    lines.append('  \\midrule')
    for _, row in df.iterrows():
        row_str = ' & '.join(str(v) for v in row.values) + ' \\\\ '
        lines.append('  ' + row_str)
    lines.append('  \\bottomrule')
    lines.append('\\end{tabular}')
    return lines


# =====
# Derived data loaders
# =====

def load_contributions():
    """Load the contributions CSV (has unnamed index column)."""
    return pd.read_csv(DERIVED_DIR / 'contributions.csv', index_col=0)


def load_behavior():
    """Load the behavior classifications CSV."""
    return pd.read_csv(DERIVED_DIR / 'behavior_classifications.csv')


def load_promises():
    """Load the promise classifications CSV."""
    return pd.read_csv(DERIVED_DIR / 'promise_classifications.csv')


def load_sentiment():
    """Load the sentiment scores CSV."""
    return pd.read_csv(DERIVED_DIR / 'sentiment_scores.csv')


def load_payoffs():
    """Load the participant payoffs CSV."""
    return pd.read_csv(DERIVED_DIR / 'participant_payoffs.csv')


# =====
# Raw data loaders
# =====

def extract_treatment(filename):
    """Extract treatment number (1 or 2) from raw filename pattern like '03_t2'."""
    match = re.search(r'_t(\d+)', filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot extract treatment from filename: {filename}")


def load_raw_data():
    """Load all raw *_data.csv files and combine with treatment column.

    For 03_t2_data.csv, only the bottom 16 rows (session_code == 'z8dowljr')
    are valid; the top 16 rows with session_code 'irrzlgk2' are dropped.
    """
    frames = []
    for path in sorted(RAW_DIR.glob('*_data.csv')):
        treatment = extract_treatment(path.name)
        df = pd.read_csv(path, encoding='utf-8-sig')
        df['treatment'] = treatment
        if path.name == _INVALID_SESSION_FILE:
            df = df[df['session.code'] == _VALID_SESSION_CODE_03]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_chat_raw():
    """Load all raw *_chat.csv files and combine with treatment column."""
    frames = []
    for path in sorted(RAW_DIR.glob('*_chat.csv')):
        treatment = extract_treatment(path.name)
        df = pd.read_csv(path, encoding='utf-8-sig')
        df['treatment'] = treatment
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
