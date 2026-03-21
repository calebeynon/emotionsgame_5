"""
Tests for round_liar_embedding_regression.py.

Tests dataset construction, model pipeline, and LaTeX output
using synthetic data with no file I/O or API calls.

Author: Claude Code
Date: 2026-03-21
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'analysis'))

from round_liar_embedding_regression import (
    _average_metrics,
    _build_latex_table,
    _compute_metrics,
    _define_models,
    _preprocess,
    cross_validate_model,
    run_model_comparison,
)


# =====
# Synthetic data builders
# =====
def _make_row(rng, i, n_dims):
    """Build one synthetic row with all required columns."""
    row = {
        'session_code': 's1', 'segment': 'supergame1',
        'round': i % 5 + 1, 'group': i % 6 + 1,
        'label': ['A', 'E', 'J', 'N'][i % 4],
        'high_contribution': rng.randint(0, 2),
        'sentiment_compound_mean': rng.rand(),
        'sentiment_positive_mean': rng.rand(),
        'sentiment_negative_mean': rng.rand(),
        'sentiment_neutral_mean': rng.rand(),
        'proj_rliar_msg_dir_small': rng.randn(),
        'proj_rliar_msg_dir_large': rng.randn(),
        'proj_rliar_pr_dir_small': rng.randn(),
        'proj_rliar_pr_dir_large': rng.randn(),
    }
    for d in range(n_dims):
        row[f'emb_{d}'] = rng.randn()
    return row


def _make_dataset(n_groups=30, n_dims=10):
    """Build a synthetic dataset DataFrame for model tests."""
    rng = np.random.RandomState(42)
    rows = [_make_row(rng, i, n_dims) for i in range(n_groups)]
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

    def test_define_models_names(self):
        """Should have the expected model names."""
        dataset = _make_dataset()
        models = _define_models(dataset)
        expected = {
            'VADER only', 'Round-liar proj. only',
            'Full embedding (PCA 50)', 'VADER + Round-liar proj.',
        }
        assert set(models.keys()) == expected

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
        dataset = _make_dataset(n_groups=40)
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

    def test_compute_metrics_keys(self):
        """_compute_metrics should return all expected keys."""
        from sklearn.linear_model import LogisticRegression
        rng = np.random.RandomState(42)
        X = rng.randn(40, 3)
        y = (X[:, 0] > 0).astype(int)
        clf = LogisticRegression(random_state=42)
        clf.fit(X[:30], y[:30])
        metrics = _compute_metrics(clf, X[30:], y[30:])
        for key in ['accuracy', 'auc', 'precision', 'recall', 'f1']:
            assert key in metrics

    def test_model_vader_only_uses_sentiment_features(self):
        """VADER only model should use 4 sentiment features."""
        dataset = _make_dataset()
        models = _define_models(dataset)
        X_vader = models['VADER only']
        assert X_vader.shape[1] == 4

    def test_model_round_liar_proj_uses_proj_columns(self):
        """Round-liar proj. model should use proj_rliar columns."""
        dataset = _make_dataset()
        models = _define_models(dataset)
        X_proj = models['Round-liar proj. only']
        assert X_proj.shape[1] >= 1

    def test_model_full_embedding_uses_emb_columns(self):
        """Full embedding model should use emb_* columns."""
        dataset = _make_dataset(n_dims=10)
        models = _define_models(dataset)
        X_emb = models['Full embedding (PCA 50)']
        assert X_emb.shape[1] == 10


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
            'model': 'Round-liar proj. only', 'accuracy': 0.8,
            'auc': 0.9, 'precision': 0.7, 'recall': 0.6, 'f1': 0.65,
        }])
        latex = _build_latex_table(results)
        assert 'Round-liar proj. only' in latex

    def test_contains_metric_values(self):
        """Output should contain formatted metric values."""
        results = pd.DataFrame([{
            'model': 'Test', 'accuracy': 0.8123, 'auc': 0.9,
            'precision': 0.7, 'recall': 0.6, 'f1': 0.65,
        }])
        latex = _build_latex_table(results)
        assert '0.8123' in latex

    def test_contains_cv_note(self):
        """Output should contain cross-validation note."""
        results = pd.DataFrame([{
            'model': 'Test', 'accuracy': 0.8, 'auc': 0.9,
            'precision': 0.7, 'recall': 0.6, 'f1': 0.65,
        }])
        latex = _build_latex_table(results)
        assert '5-fold stratified' in latex

    def test_all_four_models_in_output(self):
        """All 4 model names should appear in the LaTeX table."""
        results = pd.DataFrame([
            {'model': 'VADER only', 'accuracy': 0.7, 'auc': 0.7,
             'precision': 0.7, 'recall': 0.7, 'f1': 0.7},
            {'model': 'Round-liar proj. only', 'accuracy': 0.6,
             'auc': 0.6, 'precision': 0.6, 'recall': 0.6, 'f1': 0.6},
            {'model': 'Full embedding (PCA 50)', 'accuracy': 0.8,
             'auc': 0.8, 'precision': 0.8, 'recall': 0.8, 'f1': 0.8},
            {'model': 'VADER + Round-liar proj.', 'accuracy': 0.75,
             'auc': 0.75, 'precision': 0.75, 'recall': 0.75, 'f1': 0.75},
        ])
        latex = _build_latex_table(results)
        for name in ['VADER only', 'Round-liar proj. only',
                      'Full embedding (PCA 50)', 'VADER + Round-liar proj.']:
            assert name in latex
