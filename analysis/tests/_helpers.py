"""
Shared helpers for dynamic regression panel tests.

Provides row lookup and lag validation utilities used across
test_dynamic_regression_panel.py, test_dynamic_regression_merged_panel.py,
and test_dynamic_regression_minmedmax.py.

Author: Claude Code
Date: 2026-04-19
"""

import numpy as np
import pandas as pd


def get_row(df: pd.DataFrame, session_code: str, segment: str,
            round_num: int, label: str) -> pd.DataFrame:
    """Fetch a single row by the composite panel key."""
    mask = (
        (df['session_code'] == session_code)
        & (df['segment'] == segment)
        & (df['round'] == round_num)
        & (df['label'] == label)
    )
    result = df[mask]
    assert len(result) == 1, (
        f"Expected 1 row for ({session_code}, {segment}, {round_num}, "
        f"{label}), got {len(result)}"
    )
    return result


def validate_lag_column(df: pd.DataFrame, source_col: str,
                        lag_col: str) -> list:
    """Check that lag_col[t] == source_col[t-1] within each subject.

    Returns a list of error strings; empty list means all lags are correct.
    """
    sorted_df = df.sort_values(['subject_id', 'period'])
    expected = sorted_df.groupby('subject_id')[source_col].shift(1)
    expected[sorted_df['period'].values == 1] = np.nan
    actual = sorted_df[lag_col]
    mask = expected.notna()
    mismatches = mask & ((actual - expected).abs() > 1e-10)
    bad = sorted_df[mismatches].head(10)
    return [
        f"subject={r['subject_id']} period={r['period']}: "
        f"expected={expected.iloc[idx]}, got={r[lag_col]}"
        for idx, (_, r) in enumerate(bad.iterrows())
    ]
