"""
Tests for loud-failure guards in build_dynamic_regression_panel.py.

Validates the row-count / key-uniqueness merge guards, the NaN-bound
guard on word_count and sentiment_compound_mean, and the made_promise
NaN guard added under issue #68 (PR #70 review fixes).

Author: test-writer (pr70-fixes team)
Date: 2026-04-19
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "derived"))

from build_dynamic_regression_panel import (  # noqa: E402
    MAX_NO_MESSAGE_ROUNDS,
    MERGE_KEYS,
    convert_made_promise,
    fill_no_message_rounds,
    safe_left_merge,
)

# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / "datastore" / "derived"
DYNAMIC_PANEL_CSV = DERIVED_DIR / "dynamic_regression_panel.csv"


# =====
# Helpers — build tiny synthetic LHS/RHS DataFrames
# =====
def _make_base_df(n_rows: int = 4) -> pd.DataFrame:
    """Build a minimal LHS DataFrame with unique MERGE_KEYS."""
    return pd.DataFrame({
        "session_code": ["s1"] * n_rows,
        "segment": ["supergame1"] * n_rows,
        "round": list(range(1, n_rows + 1)),
        "label": ["A"] * n_rows,
        "contribution": [10] * n_rows,
    })


def _make_rhs_unique(n_rows: int = 4) -> pd.DataFrame:
    """RHS with unique keys matching _make_base_df()."""
    return pd.DataFrame({
        "session_code": ["s1"] * n_rows,
        "segment": ["supergame1"] * n_rows,
        "round": list(range(1, n_rows + 1)),
        "label": ["A"] * n_rows,
        "word_count": [5.0] * n_rows,
    })


def _make_panel_with_nan(
    wc_nan: int = 0, sent_nan: int = 0, n_rows: int = 10
) -> pd.DataFrame:
    """Build a synthetic merged panel with controllable NaN counts in r>1 rows."""
    df = pd.DataFrame({
        "round": [1] + [2] * (n_rows - 1),
        "word_count": [0.0] * n_rows,
        "sentiment_compound_mean": [0.0] * n_rows,
    })
    r2_idx = df.index[df["round"] > 1].tolist()
    for i in r2_idx[:wc_nan]:
        df.at[i, "word_count"] = np.nan
    for i in r2_idx[:sent_nan]:
        df.at[i, "sentiment_compound_mean"] = np.nan
    return df


# =====
# safe_left_merge — duplicate-key guard
# =====
def test_safe_left_merge_raises_on_duplicate_rhs_keys():
    """Duplicate keys in RHS must raise (pandas validate='one_to_one')."""
    base = _make_base_df(n_rows=4)
    rhs = _make_rhs_unique(n_rows=4)
    rhs = pd.concat([rhs, rhs.iloc[[0]]], ignore_index=True)
    with pytest.raises(pd.errors.MergeError, match="one-to-one|unique"):
        safe_left_merge(base, rhs, "test_rhs")


def test_safe_left_merge_raises_on_duplicate_lhs_keys():
    """Duplicate keys in LHS must also raise under one_to_one validation."""
    base = pd.concat([_make_base_df(n_rows=4), _make_base_df(n_rows=4).iloc[[0]]],
                     ignore_index=True)
    rhs = _make_rhs_unique(n_rows=4)
    with pytest.raises(pd.errors.MergeError, match="one-to-one|unique"):
        safe_left_merge(base, rhs, "test_rhs")


# =====
# safe_left_merge — row-count preserved, missing keys allowed (left join)
# =====
def test_safe_left_merge_succeeds_on_unique_keys():
    """Unique keys on both sides: merge succeeds and row count preserved."""
    base = _make_base_df(n_rows=4)
    rhs = _make_rhs_unique(n_rows=4)
    merged = safe_left_merge(base, rhs, "test_rhs")
    assert len(merged) == len(base)
    assert "word_count" in merged.columns
    assert merged["word_count"].notna().all()


def test_safe_left_merge_missing_rhs_keys_produces_nan_not_raise():
    """RHS missing keys is OK for a left join — row count preserved, NaN filled."""
    base = _make_base_df(n_rows=4)
    rhs = _make_rhs_unique(n_rows=2)
    merged = safe_left_merge(base, rhs, "test_rhs")
    assert len(merged) == len(base)
    assert merged["word_count"].isna().sum() == 2


# =====
# fill_no_message_rounds — NaN bound guard
# =====
def test_fill_no_message_rounds_within_bound_succeeds():
    """NaN count at or below MAX_NO_MESSAGE_ROUNDS should pass through."""
    bound = MAX_NO_MESSAGE_ROUNDS
    df = _make_panel_with_nan(wc_nan=bound, sent_nan=0, n_rows=bound + 2)
    filled = fill_no_message_rounds(df.copy())
    assert filled["word_count"].isna().sum() == 0


def test_fill_no_message_rounds_word_count_over_bound_raises():
    """word_count NaN exceeding the bound must raise ValueError with counts."""
    bound = MAX_NO_MESSAGE_ROUNDS
    over = bound + 1
    df = _make_panel_with_nan(wc_nan=over, sent_nan=0, n_rows=over + 2)
    expected = f"word_count NaN count {over} exceeds bound {bound}"
    with pytest.raises(ValueError, match=expected):
        fill_no_message_rounds(df)


def test_fill_no_message_rounds_sentiment_over_bound_raises():
    """sentiment_compound_mean NaN exceeding bound must raise."""
    bound = MAX_NO_MESSAGE_ROUNDS
    over = bound + 5
    df = _make_panel_with_nan(wc_nan=0, sent_nan=over, n_rows=over + 2)
    expected = f"sentiment_compound_mean NaN count {over} exceeds bound {bound}"
    with pytest.raises(ValueError, match=expected):
        fill_no_message_rounds(df)


def test_fill_no_message_rounds_round1_nan_not_counted():
    """NaN at round==1 is permitted and not counted toward the bound."""
    df = pd.DataFrame({
        "round": [1, 1, 2, 2],
        "word_count": [np.nan, np.nan, 3.0, 4.0],
        "sentiment_compound_mean": [np.nan, np.nan, 0.1, 0.2],
    })
    filled = fill_no_message_rounds(df.copy())
    # Round 1 NaNs remain (mask only fills round > 1)
    assert filled.loc[filled["round"] == 1, "word_count"].isna().sum() == 2


# =====
# convert_made_promise — NaN guard
# =====
def test_convert_made_promise_no_nan_succeeds():
    """All-boolean made_promise should convert to 0/1 ints."""
    df = pd.DataFrame({
        "session_code": ["s1", "s1"],
        "segment": ["supergame1", "supergame1"],
        "round": [1, 2],
        "label": ["A", "A"],
        "made_promise": [True, False],
    })
    out = convert_made_promise(df.copy())
    assert list(out["made_promise"]) == [1, 0]
    assert out["made_promise"].dtype.kind in ("i", "u")


def test_convert_made_promise_with_nan_raises_and_names_keys():
    """NaN in made_promise must raise and include the offending MERGE_KEYS."""
    df = pd.DataFrame({
        "session_code": ["s1", "s2"],
        "segment": ["supergame1", "supergame2"],
        "round": [1, 3],
        "label": ["A", "B"],
        "made_promise": [True, np.nan],
    })
    with pytest.raises(ValueError, match=r"made_promise has 1 NaN"):
        convert_made_promise(df)


def test_convert_made_promise_error_lists_offending_keys():
    """Error message should include session/segment/round/label identifying the bad row."""
    df = pd.DataFrame({
        "session_code": ["sA", "sB"],
        "segment": ["supergame1", "supergame2"],
        "round": [2, 4],
        "label": ["C", "D"],
        "made_promise": [np.nan, True],
    })
    with pytest.raises(ValueError) as exc:
        convert_made_promise(df)
    msg = str(exc.value)
    assert "sA" in msg
    assert "supergame1" in msg
    assert "C" in msg
    # MERGE_KEYS order preserved in to_dict payload
    for key in MERGE_KEYS:
        assert key in msg


# =====
# Regression against real data — panel CSV still loadable and guards don't fire
# =====
def test_pipeline_safe_left_merge_on_real_keys():
    """safe_left_merge should succeed when applied to real panel keys (no dupes)."""
    if not DYNAMIC_PANEL_CSV.exists():
        pytest.skip(f"dynamic_regression_panel.csv missing: {DYNAMIC_PANEL_CSV}")
    df = pd.read_csv(DYNAMIC_PANEL_CSV)
    left = df[MERGE_KEYS].copy()
    right = df[MERGE_KEYS + ["made_promise"]].copy()
    merged = safe_left_merge(left, right, "real_panel")
    assert len(merged) == len(left)


def test_real_panel_word_count_nan_under_bound():
    """Observed word_count NaN count should be well below MAX_NO_MESSAGE_ROUNDS."""
    if not DYNAMIC_PANEL_CSV.exists():
        pytest.skip(f"dynamic_regression_panel.csv missing: {DYNAMIC_PANEL_CSV}")
    df = pd.read_csv(DYNAMIC_PANEL_CSV)
    r2_nan = df.loc[df["round"] > 1, "word_count"].isna().sum()
    assert r2_nan <= MAX_NO_MESSAGE_ROUNDS, (
        f"Observed {r2_nan} NaN exceeds bound {MAX_NO_MESSAGE_ROUNDS}; "
        "either upstream regressed or the bound needs revisiting."
    )
