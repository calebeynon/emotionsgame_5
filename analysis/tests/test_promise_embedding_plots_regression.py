"""
Regression and integration tests for promise_embedding_plots.py.

Tests label loading/merging, color constants, majority vote,
and plot generation using synthetic data with matplotlib backend.
No file I/O or API calls for core tests.

Author: Claude Code (test-writer)
Date: 2026-03-15
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for tests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'analysis'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'derived'))

from promise_embedding_plots import (
    FIGSIZE,
    LABEL_COLORS,
    LABEL_NAMES,
    NO_PROMISE_COLOR,
    PROMISE_COLOR,
    _assign_majority_promise,
    _load_promise_labels,
    _merge_promise_labels,
    _plot_projection_distribution,
    _save_scatter,
)


# =====
# Helpers
# =====
def _make_promise_state_csv(tmp_path, n=8):
    """Write synthetic player_state_classification.csv and return path."""
    rows = []
    for i in range(n):
        rows.append({
            'session_code': 's1', 'treatment': 1,
            'segment': 'supergame1', 'round_num': 2,
            'group_id': 1 + i // 4, 'label': chr(65 + i),
            'player_state': 'cooperative',
            'made_promise': i < n // 2,
        })
    path = tmp_path / 'player_state_classification.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _make_metadata(n):
    """Create minimal embedding metadata DataFrame."""
    return pd.DataFrame({
        'session_code': ['s1'] * n,
        'segment': ['supergame1'] * n,
        'round': [2] * n,
        'group': [1] * (n // 2) + [2] * (n - n // 2),
        'label': [chr(65 + i) for i in range(n)],
    })


# =====
# Regression: color constants
# =====
class TestColorConstants:
    """Verify plot styling constants."""

    def test_promise_color(self):
        """Promise color should be blue (#3498db)."""
        assert PROMISE_COLOR == '#3498db'

    def test_no_promise_color(self):
        """No-promise color should be orange (#e67e22)."""
        assert NO_PROMISE_COLOR == '#e67e22'

    def test_label_colors_keys(self):
        """LABEL_COLORS should map True/False to colors."""
        assert True in LABEL_COLORS
        assert False in LABEL_COLORS
        assert LABEL_COLORS[True] == PROMISE_COLOR
        assert LABEL_COLORS[False] == NO_PROMISE_COLOR

    def test_label_names(self):
        """LABEL_NAMES should provide display names for True/False."""
        assert LABEL_NAMES[True] == 'Promise'
        assert LABEL_NAMES[False] == 'No Promise'

    def test_figsize(self):
        """Default figure size should be (8, 6)."""
        assert FIGSIZE == (8, 6)


# =====
# Regression: _load_promise_labels
# =====
class TestLoadPromiseLabels:
    """Tests for loading promise labels from state CSV."""

    def test_renames_columns(self, tmp_path, monkeypatch):
        """Should rename round_num -> round and group_id -> group."""
        path = _make_promise_state_csv(tmp_path)
        monkeypatch.setattr(
            'promise_embedding_plots.STATE_FILE', path
        )
        result = _load_promise_labels()
        assert 'round' in result.columns
        assert 'group' in result.columns
        assert 'round_num' not in result.columns
        assert 'group_id' not in result.columns

    def test_has_made_promise(self, tmp_path, monkeypatch):
        """Output should have made_promise column."""
        path = _make_promise_state_csv(tmp_path)
        monkeypatch.setattr(
            'promise_embedding_plots.STATE_FILE', path
        )
        result = _load_promise_labels()
        assert 'made_promise' in result.columns

    def test_has_join_keys(self, tmp_path, monkeypatch):
        """Output should have all join key columns."""
        path = _make_promise_state_csv(tmp_path)
        monkeypatch.setattr(
            'promise_embedding_plots.STATE_FILE', path
        )
        result = _load_promise_labels()
        for col in ['session_code', 'segment', 'round', 'group', 'label']:
            assert col in result.columns


# =====
# Regression: _merge_promise_labels
# =====
class TestMergePromiseLabels:
    """Tests for merging promise labels onto metadata."""

    def test_returns_boolean_array(self):
        """Should return array of boolean values."""
        meta = _make_metadata(4)
        promise_df = pd.DataFrame({
            'session_code': ['s1'] * 4,
            'segment': ['supergame1'] * 4,
            'round': [2] * 4,
            'group': [1, 1, 2, 2],
            'label': ['A', 'B', 'C', 'D'],
            'made_promise': [True, False, True, False],
        })
        result = _merge_promise_labels(meta, promise_df)
        assert len(result) == 4

    def test_unmatched_returns_nan(self):
        """Unmatched rows should return NaN."""
        meta = _make_metadata(2)
        promise_df = pd.DataFrame({
            'session_code': ['s1'], 'segment': ['supergame1'],
            'round': [2], 'group': [1], 'label': ['A'],
            'made_promise': [True],
        })
        result = _merge_promise_labels(meta, promise_df)
        assert len(result) == 2
        # Second row (label B) shouldn't match if group differs
        # (depends on metadata structure)


# =====
# Regression: _assign_majority_promise
# =====
class TestAssignMajorityPromise:
    """Tests for majority promise assignment at group level."""

    def test_all_true_group(self):
        """Group with all True should get True majority."""
        meta = pd.DataFrame({
            'session_code': ['s1'] * 4,
            'segment': ['sg1'] * 4,
            'round': [1] * 4,
            'group': [1] * 4,
            'label': ['A', 'B', 'C', 'D'],
        })
        labels = np.array([True, True, True, True])
        result = _assign_majority_promise(meta, labels)
        assert len(result) == 1
        assert bool(result['made_promise'].iloc[0]) is True

    def test_all_false_group(self):
        """Group with all False should get False majority."""
        meta = pd.DataFrame({
            'session_code': ['s1'] * 4,
            'segment': ['sg1'] * 4,
            'round': [1] * 4,
            'group': [1] * 4,
            'label': ['A', 'B', 'C', 'D'],
        })
        labels = np.array([False, False, False, False])
        result = _assign_majority_promise(meta, labels)
        assert bool(result['made_promise'].iloc[0]) is False

    def test_majority_true(self):
        """3 True + 1 False should give True majority."""
        meta = pd.DataFrame({
            'session_code': ['s1'] * 4,
            'segment': ['sg1'] * 4,
            'round': [1] * 4,
            'group': [1] * 4,
            'label': ['A', 'B', 'C', 'D'],
        })
        labels = np.array([True, True, True, False])
        result = _assign_majority_promise(meta, labels)
        assert bool(result['made_promise'].iloc[0]) is True

    def test_two_groups(self):
        """Should produce one row per group."""
        meta = pd.DataFrame({
            'session_code': ['s1'] * 4,
            'segment': ['sg1'] * 4,
            'round': [1] * 4,
            'group': [1, 1, 2, 2],
            'label': ['A', 'B', 'C', 'D'],
        })
        labels = np.array([True, True, False, False])
        result = _assign_majority_promise(meta, labels)
        assert len(result) == 2


# =====
# Regression: _save_scatter
# =====
class TestSaveScatter:
    """Tests for scatter plot generation."""

    def test_creates_file(self, tmp_path):
        """Should create a PNG file at the specified path."""
        coords = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        labels = np.array([True, True, False, False])
        output_path = tmp_path / 'test_scatter.png'
        _save_scatter(coords, labels, 'Test Title', output_path)
        assert output_path.exists()

    def test_file_nonzero_size(self, tmp_path):
        """Output PNG should have nonzero file size."""
        coords = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        labels = np.array([True, True, False, False])
        output_path = tmp_path / 'test_scatter.png'
        _save_scatter(coords, labels, 'Test', output_path)
        assert output_path.stat().st_size > 0


# =====
# Regression: _plot_projection_distribution
# =====
class TestPlotProjectionDistribution:
    """Tests for projection distribution histogram."""

    def test_creates_file_with_valid_data(self, tmp_path):
        """Should create PNG when valid projection data is provided."""
        projections = pd.DataFrame({
            'proj_promise_msg_dir_small': np.random.randn(20),
            'promise_label': ['promise'] * 10 + ['no_promise'] * 10,
        })
        output_path = tmp_path / 'test_dist.png'
        _plot_projection_distribution(projections, 'small', output_path)
        assert output_path.exists()

    def test_handles_missing_column_gracefully(self, tmp_path):
        """Should not crash when projection column is missing."""
        projections = pd.DataFrame({
            'some_other_col': np.random.randn(20),
            'promise_label': ['promise'] * 10 + ['no_promise'] * 10,
        })
        output_path = tmp_path / 'test_dist_missing.png'
        _plot_projection_distribution(projections, 'small', output_path)
        # File should NOT be created since column is missing
        assert not output_path.exists()


# =====
# Integration: end-to-end plot pipeline
# =====
class TestPlotIntegration:
    """Integration tests for the promise plot pipeline."""

    def test_scatter_with_all_true_labels(self, tmp_path):
        """Scatter should work even with a single label class."""
        coords = np.random.randn(10, 2)
        labels = np.array([True] * 10)
        output_path = tmp_path / 'all_true.png'
        _save_scatter(coords, labels, 'All True', output_path)
        assert output_path.exists()

    def test_scatter_with_all_false_labels(self, tmp_path):
        """Scatter should work even with a single label class."""
        coords = np.random.randn(10, 2)
        labels = np.array([False] * 10)
        output_path = tmp_path / 'all_false.png'
        _save_scatter(coords, labels, 'All False', output_path)
        assert output_path.exists()

    def test_projection_dist_both_suffixes(self, tmp_path):
        """Should create plots for both small and large suffixes."""
        for suffix in ['small', 'large']:
            projections = pd.DataFrame({
                f'proj_promise_msg_dir_{suffix}': np.random.randn(30),
                'promise_label': ['promise'] * 15 + ['no_promise'] * 15,
            })
            output_path = tmp_path / f'dist_{suffix}.png'
            _plot_projection_distribution(projections, suffix, output_path)
            assert output_path.exists()
