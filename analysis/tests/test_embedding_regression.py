"""
Tests for embedding_regression.py.

Tests dataset construction, aggregation, and model pipeline
using synthetic data with no file I/O or API calls.

Author: Claude Code
Date: 2026-03-15
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'analysis'))

from embedding_regression import (
    _average_metrics,
    _build_latex_table,
    _compute_metrics,
    _define_models,
    _preprocess,
    cross_validate_model,
    run_model_comparison,
)



def _make_dataset(n_rows=30, n_dims=10):
    """Build a synthetic dataset DataFrame for model tests."""
    rng = np.random.RandomState(42)
    rows = []
    for i in range(n_rows):
        row = {
            'session_code': 's1', 'segment': 'supergame1',
            'round': i % 5 + 1, 'group': i % 6 + 1,
            'label': chr(65 + i % 16),
            'high_contribution': rng.randint(0, 2),
            'sentiment_compound_mean': rng.rand(),
            'sentiment_positive_mean': rng.rand(),
            'sentiment_negative_mean': rng.rand(),
            'sentiment_neutral_mean': rng.rand(),
            'proj_pr_dir_small': rng.randn(),
            'proj_pr_dir_large': rng.randn(),
        }
        for d in range(n_dims):
            row[f'emb_{d}'] = rng.randn()
        rows.append(row)
    return pd.DataFrame(rows)


# =====
# Tests for model pipeline
# =====
class TestModelPipeline:
    """Tests for CV and metrics computation."""

    def test_define_models_returns_four(self):
        """Should define exactly 4 model configurations."""
        dataset = _make_dataset()
        models = _define_models(dataset)
        assert len(models) == 4

    def test_cross_validate_returns_all_metrics(self):
        """CV should return accuracy, auc, precision, recall, f1."""
        rng = np.random.RandomState(0)
        X = rng.randn(40, 3)
        y = (X[:, 0] > 0).astype(int)
        result = cross_validate_model(X, y)

        for key in ['accuracy', 'auc', 'precision', 'recall', 'f1']:
            assert key in result
            assert 0.0 <= result[key] <= 1.0

    def test_preprocess_scales_data(self):
        """Preprocessed training data should be approximately standardized."""
        rng = np.random.RandomState(0)
        X = rng.randn(50, 5) * 100 + 50
        X_train, X_test = _preprocess(X[:40], X[40:], use_pca=False)

        assert np.abs(X_train.mean()) < 0.1
        assert np.abs(X_train.std() - 1.0) < 0.1

    def test_preprocess_pca_reduces_dims(self):
        """PCA preprocessing should reduce dimensionality."""
        rng = np.random.RandomState(0)
        X = rng.randn(60, 100)
        X_train, X_test = _preprocess(X[:50], X[50:], use_pca=True)

        assert X_train.shape[1] == 50
        assert X_test.shape[1] == 50

    def test_run_model_comparison_shape(self):
        """Comparison should return 4 rows with model and metric cols."""
        dataset = _make_dataset(n_rows=40)
        results = run_model_comparison(dataset)

        assert len(results) == 4
        assert 'model' in results.columns
        assert 'accuracy' in results.columns

    def test_average_metrics(self):
        """Averaging two fold dicts should produce correct means."""
        folds = [
            {'accuracy': 0.8, 'auc': 0.9, 'precision': 0.7,
             'recall': 0.6, 'f1': 0.65},
            {'accuracy': 0.6, 'auc': 0.7, 'precision': 0.5,
             'recall': 0.4, 'f1': 0.45},
        ]
        result = _average_metrics(folds)
        assert result['accuracy'] == pytest.approx(0.7)
        assert result['auc'] == pytest.approx(0.8)


# =====
# Tests for LaTeX output
# =====
class TestLatexOutput:
    """Tests for LaTeX table generation."""

    def test_contains_tabular(self):
        """Output should contain LaTeX tabular environment."""
        results = pd.DataFrame([{
            'model': 'Test', 'accuracy': 0.8, 'auc': 0.9,
            'precision': 0.7, 'recall': 0.6, 'f1': 0.65,
        }])
        latex = _build_latex_table(results)
        assert r'\begin{tabular}' in latex
        assert r'\end{tabular}' in latex

    def test_contains_model_name(self):
        """Output should contain the model name."""
        results = pd.DataFrame([{
            'model': 'VADER only', 'accuracy': 0.8, 'auc': 0.9,
            'precision': 0.7, 'recall': 0.6, 'f1': 0.65,
        }])
        latex = _build_latex_table(results)
        assert 'VADER only' in latex

    def test_contains_metric_values(self):
        """Output should contain formatted metric values."""
        results = pd.DataFrame([{
            'model': 'Test', 'accuracy': 0.8123, 'auc': 0.9,
            'precision': 0.7, 'recall': 0.6, 'f1': 0.65,
        }])
        latex = _build_latex_table(results)
        assert '0.8123' in latex
