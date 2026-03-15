"""
Tests for promise_embedding_regression.py.

Tests dataset construction, promise majority vote, model pipeline,
and LaTeX output using synthetic data with no file I/O or API calls.

Author: Claude Code
Date: 2026-03-15
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'analysis'))

from promise_embedding_regression import (
    _aggregate_promise_labels,
    _aggregate_promise_projections,
    _aggregate_sentiment,
    _average_metrics,
    _build_latex_table,
    _compute_metrics,
    _define_models,
    _preprocess,
    _promise_majority_vote,
    build_dataset,
    cross_validate_model,
    run_model_comparison,
)


# =====
# Synthetic data builders
# =====
def _make_state_csv(tmp_path):
    """Write synthetic state CSV with made_promise column and return path."""
    rows = [
        {'session_code': 's1', 'treatment': 1, 'segment': 'supergame1',
         'round_num': 2, 'group_id': 1, 'label': 'A',
         'made_promise': True},
        {'session_code': 's1', 'treatment': 1, 'segment': 'supergame1',
         'round_num': 2, 'group_id': 1, 'label': 'E',
         'made_promise': True},
        {'session_code': 's1', 'treatment': 1, 'segment': 'supergame1',
         'round_num': 2, 'group_id': 1, 'label': 'J',
         'made_promise': False},
        {'session_code': 's1', 'treatment': 1, 'segment': 'supergame1',
         'round_num': 2, 'group_id': 1, 'label': 'N',
         'made_promise': True},
    ]
    path = tmp_path / 'state.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _make_sentiment_csv(tmp_path):
    """Write synthetic sentiment CSV and return path."""
    rows = []
    for label in ['A', 'E', 'J', 'N']:
        rows.append({
            'session_code': 's1', 'treatment': 1,
            'segment': 'supergame1', 'round': 2, 'group': 1,
            'label': label,
            'sentiment_compound_mean': 0.5,
            'sentiment_positive_mean': 0.3,
            'sentiment_negative_mean': 0.1,
            'sentiment_neutral_mean': 0.6,
        })
    path = tmp_path / 'sentiment.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _make_promise_projections_csv(tmp_path):
    """Write synthetic promise projections CSV and return path."""
    rows = []
    for label in ['A', 'E', 'J', 'N']:
        rows.append({
            'session_code': 's1', 'segment': 'supergame1',
            'round': 2, 'group': 1, 'label': label,
            'proj_promise_msg_dir_small': 0.8,
            'proj_promise_msg_dir_large': 0.7,
        })
    path = tmp_path / 'promise_projections.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _make_embeddings_parquet(tmp_path, n_dims=10):
    """Write synthetic embeddings parquet and return path."""
    rows = []
    for label in ['A', 'E', 'J', 'N']:
        row = {
            'session_code': 's1', 'treatment': 1,
            'segment': 'supergame1', 'round': 2, 'group': 1,
            'label': label, 'player_state': 'cooperative',
        }
        for i in range(n_dims):
            row[f'emb_{i}'] = np.random.rand()
        rows.append(row)
    path = tmp_path / 'embeddings.parquet'
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _make_dataset(n_groups=30, n_dims=10):
    """Build a synthetic dataset DataFrame for model tests."""
    rng = np.random.RandomState(42)
    rows = []
    for i in range(n_groups):
        row = {
            'session_code': 's1', 'segment': 'supergame1',
            'round': i % 5 + 1, 'group': i % 6 + 1,
            'made_promise': rng.randint(0, 2),
            'sentiment_compound_mean': rng.rand(),
            'sentiment_positive_mean': rng.rand(),
            'sentiment_negative_mean': rng.rand(),
            'sentiment_neutral_mean': rng.rand(),
            'proj_promise_msg_dir_small': rng.randn(),
            'proj_promise_msg_dir_large': rng.randn(),
        }
        for d in range(n_dims):
            row[f'emb_{d}'] = rng.randn()
        rows.append(row)
    return pd.DataFrame(rows)


# =====
# Tests for _promise_majority_vote
# =====
class TestPromiseMajorityVote:
    """Tests for promise majority vote aggregation."""

    def test_clear_majority_true(self):
        """Three True, one False -> True."""
        values = pd.Series([True, True, True, False])
        assert _promise_majority_vote(values) is True

    def test_tie_goes_to_true(self):
        """Two True, two False -> True (tie rule)."""
        values = pd.Series([True, False, True, False])
        assert _promise_majority_vote(values) is True

    def test_clear_majority_false(self):
        """Three False, one True -> False."""
        values = pd.Series([False, False, False, True])
        assert _promise_majority_vote(values) is False

    def test_all_true(self):
        """All True -> True."""
        values = pd.Series([True, True, True, True])
        assert _promise_majority_vote(values) is True

    def test_all_false(self):
        """All False -> False."""
        values = pd.Series([False, False, False, False])
        assert _promise_majority_vote(values) is False


# =====
# Tests for aggregation functions
# =====
class TestAggregation:
    """Tests for data aggregation helpers."""

    def test_aggregate_promise_renames_columns(self, tmp_path):
        """Promise aggregation should rename round_num/group_id."""
        path = _make_state_csv(tmp_path)
        result = _aggregate_promise_labels(path)
        assert 'round' in result.columns
        assert 'group' in result.columns
        assert 'round_num' not in result.columns

    def test_aggregate_promise_produces_binary(self, tmp_path):
        """Output should have binary made_promise column."""
        path = _make_state_csv(tmp_path)
        result = _aggregate_promise_labels(path)
        assert set(result['made_promise'].unique()).issubset({0, 1})

    def test_aggregate_promise_one_row_per_group(self, tmp_path):
        """4 players in one group should produce 1 row."""
        path = _make_state_csv(tmp_path)
        result = _aggregate_promise_labels(path)
        assert len(result) == 1

    def test_aggregate_promise_majority_value(self, tmp_path):
        """3 True, 1 False should aggregate to 1 (True)."""
        path = _make_state_csv(tmp_path)
        result = _aggregate_promise_labels(path)
        assert result['made_promise'].iloc[0] == 1

    def test_aggregate_sentiment_means(self, tmp_path):
        """Sentiment should be averaged per group-round."""
        path = _make_sentiment_csv(tmp_path)
        result = _aggregate_sentiment(path)
        assert len(result) == 1
        assert result['sentiment_compound_mean'].iloc[0] == pytest.approx(0.5)

    def test_aggregate_promise_projections(self, tmp_path):
        """Promise projections should be averaged per group-round."""
        path = _make_promise_projections_csv(tmp_path)
        result = _aggregate_promise_projections(path)
        assert len(result) == 1
        assert 'proj_promise_msg_dir_small' in result.columns


# =====
# Tests for build_dataset
# =====
class TestBuildDataset:
    """Tests for full dataset merge."""

    def test_produces_all_required_columns(self, tmp_path):
        """Merged dataset must have VADER, embedding, projection, and target."""
        emb = _make_embeddings_parquet(tmp_path)
        sent = _make_sentiment_csv(tmp_path)
        state = _make_state_csv(tmp_path)
        proj = _make_promise_projections_csv(tmp_path)

        result = build_dataset(emb, sent, state, proj)

        assert 'made_promise' in result.columns
        assert 'sentiment_compound_mean' in result.columns
        assert 'proj_promise_msg_dir_small' in result.columns
        assert any(c.startswith('emb_') for c in result.columns)

    def test_one_row_per_group_round(self, tmp_path):
        """One group with 4 players -> 1 row."""
        emb = _make_embeddings_parquet(tmp_path)
        sent = _make_sentiment_csv(tmp_path)
        state = _make_state_csv(tmp_path)
        proj = _make_promise_projections_csv(tmp_path)

        result = build_dataset(emb, sent, state, proj)
        assert len(result) == 1


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
            'VADER only', 'Promise projection only',
            'Full embedding (PCA 50)', 'VADER + Promise proj.',
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
            'model': 'Promise projection only', 'accuracy': 0.8,
            'auc': 0.9, 'precision': 0.7, 'recall': 0.6, 'f1': 0.65,
        }])
        latex = _build_latex_table(results)
        assert 'Promise projection only' in latex

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
        assert '5-fold stratified cross-validation' in latex
