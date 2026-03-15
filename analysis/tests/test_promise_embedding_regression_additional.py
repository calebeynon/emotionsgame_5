"""
Regression and integration tests for promise_embedding_regression.py.

Covers majority vote, dataset construction, model pipeline,
cross-validation, and LaTeX output with synthetic data.
No file I/O or API calls.

Author: Claude Code (test-writer)
Date: 2026-03-15
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'analysis'))

from promise_embedding_regression import (
    GROUP_KEYS,
    N_FOLDS,
    PCA_COMPONENTS,
    VADER_FEATURES,
    _aggregate_promise_projections,
    _average_metrics,
    _build_latex_table,
    _define_models,
    _preprocess,
    _promise_majority_vote,
    cross_validate_model,
    run_model_comparison,
)


# =====
# Helpers
# =====
def _make_dataset(n_groups=40, n_dims=10, seed=42):
    """Build synthetic dataset with promise labels and projections."""
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_groups):
        made_promise = i % 2
        sign = 1 if made_promise else -1
        row = {
            'session_code': f's{i % 3}',
            'segment': f'supergame{i % 5 + 1}',
            'round': i % 5 + 1,
            'group': i % 6 + 1,
            'made_promise': made_promise,
            'sentiment_compound_mean': sign * 0.5 + rng.randn() * 0.1,
            'sentiment_positive_mean': max(0, sign * 0.3 + rng.randn() * 0.1),
            'sentiment_negative_mean': max(0, -sign * 0.3 + rng.randn() * 0.1),
            'sentiment_neutral_mean': 0.5 + rng.randn() * 0.1,
            'proj_promise_msg_dir_small': sign * 1.0 + rng.randn() * 0.2,
            'proj_promise_pr_dir_small': sign * 0.8 + rng.randn() * 0.2,
            'proj_promise_msg_dir_large': sign * 1.5 + rng.randn() * 0.3,
            'proj_promise_pr_dir_large': sign * 1.2 + rng.randn() * 0.3,
        }
        for d in range(n_dims):
            row[f'emb_{d}'] = sign * rng.rand() + rng.randn() * 0.1
        rows.append(row)
    return pd.DataFrame(rows)


def _make_promise_projections_csv(tmp_path):
    """Write synthetic promise projections CSV and return path."""
    rows = []
    for label in ['A', 'E', 'J', 'N']:
        rows.append({
            'session_code': 's1', 'segment': 'supergame1',
            'round': 2, 'group': 1, 'label': label,
            'proj_promise_msg_dir_small': 0.5,
            'proj_promise_pr_dir_small': 0.4,
            'proj_promise_msg_dir_large': 0.6,
            'proj_promise_pr_dir_large': 0.3,
        })
    path = tmp_path / 'promise_projections.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# =====
# Regression: constants
# =====
class TestConstantsRegression:
    """Verify module constants match expected values."""

    def test_n_folds(self):
        """Should use 5-fold cross-validation."""
        assert N_FOLDS == 5

    def test_pca_components(self):
        """PCA should reduce to 50 components."""
        assert PCA_COMPONENTS == 50

    def test_group_keys(self):
        """Group keys should match expected list."""
        expected = ['session_code', 'segment', 'round', 'group']
        assert GROUP_KEYS == expected

    def test_vader_features_count(self):
        """VADER feature list should have 4 items."""
        assert len(VADER_FEATURES) == 4
        assert 'sentiment_compound_mean' in VADER_FEATURES


# =====
# Regression: _promise_majority_vote
# =====
class TestPromiseMajorityVote:
    """Tests for promise-specific majority vote."""

    def test_all_true(self):
        """All True should return True."""
        assert _promise_majority_vote(pd.Series([True, True, True, True])) is True

    def test_all_false(self):
        """All False should return False."""
        assert _promise_majority_vote(pd.Series([False, False, False, False])) is False

    def test_majority_true(self):
        """Three True, one False should return True."""
        assert _promise_majority_vote(pd.Series([True, True, True, False])) is True

    def test_majority_false(self):
        """Three False, one True should return False."""
        assert _promise_majority_vote(pd.Series([False, False, False, True])) is False

    def test_tie_goes_true(self):
        """Two True, two False should return True (tie rule)."""
        assert _promise_majority_vote(pd.Series([True, False, True, False])) is True

    def test_single_true(self):
        """Single True should return True."""
        assert _promise_majority_vote(pd.Series([True])) is True

    def test_single_false(self):
        """Single False should return False."""
        assert _promise_majority_vote(pd.Series([False])) is False


# =====
# Regression: _aggregate_promise_projections
# =====
class TestAggregatePromiseProjections:
    """Tests for promise projection aggregation."""

    def test_aggregates_to_group_round(self, tmp_path):
        """Should produce one row per group-round."""
        path = _make_promise_projections_csv(tmp_path)
        result = _aggregate_promise_projections(path)
        assert len(result) == 1

    def test_has_all_projection_columns(self, tmp_path):
        """Output should have all 4 promise projection columns."""
        path = _make_promise_projections_csv(tmp_path)
        result = _aggregate_promise_projections(path)
        for col in [
            'proj_promise_msg_dir_small', 'proj_promise_pr_dir_small',
            'proj_promise_msg_dir_large', 'proj_promise_pr_dir_large',
        ]:
            assert col in result.columns

    def test_values_are_mean(self, tmp_path):
        """Aggregated values should be the mean of input values."""
        path = _make_promise_projections_csv(tmp_path)
        result = _aggregate_promise_projections(path)
        assert result['proj_promise_msg_dir_small'].iloc[0] == pytest.approx(0.5)


# =====
# Regression: _define_models
# =====
class TestDefineModels:
    """Tests for model definition function."""

    def test_returns_four_models(self):
        """Should define exactly 4 model configurations."""
        dataset = _make_dataset()
        models = _define_models(dataset)
        assert len(models) == 4

    def test_model_names(self):
        """Should define models with expected names."""
        dataset = _make_dataset()
        models = _define_models(dataset)
        expected_names = {
            'VADER only', 'Promise projection only',
            'Full embedding (PCA 50)', 'VADER + Promise proj.',
        }
        assert set(models.keys()) == expected_names

    def test_vader_only_shape(self):
        """VADER model should have 4 features."""
        dataset = _make_dataset(n_groups=20)
        models = _define_models(dataset)
        assert models['VADER only'].shape[1] == 4

    def test_promise_projection_shape(self):
        """Promise projection model should have 4 features."""
        dataset = _make_dataset(n_groups=20)
        models = _define_models(dataset)
        assert models['Promise projection only'].shape[1] == 4

    def test_combined_shape(self):
        """Combined model should have VADER(4) + promise proj(4) = 8."""
        dataset = _make_dataset(n_groups=20)
        models = _define_models(dataset)
        assert models['VADER + Promise proj.'].shape[1] == 8

    def test_full_embedding_shape(self):
        """Full embedding should have n_dims features."""
        n_dims = 10
        dataset = _make_dataset(n_groups=20, n_dims=n_dims)
        models = _define_models(dataset)
        assert models['Full embedding (PCA 50)'].shape[1] == n_dims


# =====
# Regression: _preprocess
# =====
class TestPreprocessRegression:
    """Regression tests for preprocessing."""

    def test_pca_caps_at_sample_count(self):
        """PCA components should not exceed sample count."""
        rng = np.random.RandomState(0)
        X_train = rng.randn(30, 200)
        X_test = rng.randn(10, 200)
        X_train_out, X_test_out = _preprocess(X_train, X_test, use_pca=True)

        # min(PCA_COMPONENTS=50, 200 dims, 30 samples) = 30
        assert X_train_out.shape[1] == 30
        assert X_test_out.shape[1] == 30

    def test_no_pca_preserves_dims(self):
        """Without PCA, dimensionality should be preserved."""
        rng = np.random.RandomState(0)
        X = rng.randn(50, 8)
        X_train_out, _ = _preprocess(X[:40], X[40:], use_pca=False)
        assert X_train_out.shape[1] == 8

    def test_standardization_zero_mean(self):
        """Standardized train data should have near-zero mean per feature."""
        rng = np.random.RandomState(0)
        X = rng.randn(100, 5) * 10 + 100
        X_train, _ = _preprocess(X[:80], X[80:], use_pca=False)
        assert np.abs(X_train.mean(axis=0)).max() < 0.01


# =====
# Regression: cross_validate_model
# =====
class TestCrossValidateModelRegression:
    """Regression tests for cross-validation."""

    def test_metrics_in_valid_range(self):
        """All metrics should be in [0, 1]."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 4)
        y = (X[:, 0] > 0).astype(int)
        result = cross_validate_model(X, y)

        for key in ['accuracy', 'auc', 'precision', 'recall', 'f1']:
            assert 0.0 <= result[key] <= 1.0, f"{key} out of range"

    def test_separable_data_high_accuracy(self):
        """Clearly separable data should achieve high accuracy."""
        rng = np.random.RandomState(42)
        X_pos = rng.normal(loc=5.0, scale=0.1, size=(30, 3))
        X_neg = rng.normal(loc=-5.0, scale=0.1, size=(30, 3))
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * 30 + [0] * 30)
        result = cross_validate_model(X, y)

        assert result['accuracy'] > 0.9
        assert result['auc'] > 0.9

    def test_random_data_near_chance(self):
        """Random labels should give near-chance accuracy."""
        rng = np.random.RandomState(42)
        X = rng.randn(60, 4)
        y = rng.randint(0, 2, size=60)
        result = cross_validate_model(X, y)

        assert 0.25 <= result['accuracy'] <= 0.75

    def test_deterministic_output(self):
        """Same input should produce same results."""
        rng = np.random.RandomState(42)
        X = rng.randn(60, 4)
        y = (X[:, 0] > 0).astype(int)
        r1 = cross_validate_model(X, y)
        r2 = cross_validate_model(X, y)

        for key in ['accuracy', 'auc', 'f1']:
            assert r1[key] == pytest.approx(r2[key])


# =====
# Regression: _average_metrics
# =====
class TestAverageMetrics:
    """Regression tests for metric averaging."""

    def test_single_fold(self):
        """Single fold should return unchanged metrics."""
        folds = [{'accuracy': 0.9, 'auc': 0.95, 'precision': 0.85,
                  'recall': 0.8, 'f1': 0.82}]
        result = _average_metrics(folds)
        assert result['accuracy'] == 0.9

    def test_three_folds(self):
        """Three folds should produce correct mean."""
        folds = [
            {'accuracy': 0.7, 'auc': 0.8, 'precision': 0.6,
             'recall': 0.5, 'f1': 0.55},
            {'accuracy': 0.8, 'auc': 0.9, 'precision': 0.7,
             'recall': 0.6, 'f1': 0.65},
            {'accuracy': 0.9, 'auc': 1.0, 'precision': 0.8,
             'recall': 0.7, 'f1': 0.75},
        ]
        result = _average_metrics(folds)
        assert result['accuracy'] == pytest.approx(0.8)
        assert result['f1'] == pytest.approx(0.65)


# =====
# Integration: full model comparison pipeline
# =====
class TestFullPipeline:
    """Integration tests for the complete promise model comparison."""

    def test_run_comparison_produces_four_models(self):
        """Should compare exactly 4 models."""
        dataset = _make_dataset(n_groups=40)
        results = run_model_comparison(dataset)
        assert len(results) == 4

    def test_all_metrics_present(self):
        """Each model should have all 5 metrics."""
        dataset = _make_dataset(n_groups=40)
        results = run_model_comparison(dataset)
        for _, row in results.iterrows():
            for metric in ['accuracy', 'auc', 'precision', 'recall', 'f1']:
                assert metric in row.index
                assert 0.0 <= row[metric] <= 1.0

    def test_results_are_deterministic(self):
        """Same input should produce same results (fixed random_state)."""
        dataset = _make_dataset(n_groups=40, seed=99)
        r1 = run_model_comparison(dataset)
        r2 = run_model_comparison(dataset)

        for metric in ['accuracy', 'auc', 'f1']:
            np.testing.assert_array_almost_equal(
                r1[metric].values, r2[metric].values
            )

    def test_model_column_present(self):
        """Results should have model name column."""
        dataset = _make_dataset(n_groups=40)
        results = run_model_comparison(dataset)
        assert 'model' in results.columns


# =====
# Regression: LaTeX output
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

    def test_multiple_models(self):
        """All model names should appear in LaTeX output."""
        results = pd.DataFrame([
            {'model': 'VADER only', 'accuracy': 0.7, 'auc': 0.8,
             'precision': 0.6, 'recall': 0.5, 'f1': 0.55},
            {'model': 'VADER + Promise proj.', 'accuracy': 0.75,
             'auc': 0.85, 'precision': 0.65, 'recall': 0.55, 'f1': 0.6},
        ])
        latex = _build_latex_table(results)
        assert 'VADER only' in latex
        assert 'VADER + Promise proj.' in latex

    def test_cv_note_in_footer(self):
        """LaTeX output should mention cross-validation in footer."""
        results = pd.DataFrame([{
            'model': 'Test', 'accuracy': 0.8, 'auc': 0.9,
            'precision': 0.7, 'recall': 0.6, 'f1': 0.65,
        }])
        latex = _build_latex_table(results)
        assert 'cross-validation' in latex

    def test_contains_metric_values(self):
        """Output should contain formatted metric values."""
        results = pd.DataFrame([{
            'model': 'Test', 'accuracy': 0.8123, 'auc': 0.9,
            'precision': 0.7, 'recall': 0.6, 'f1': 0.65,
        }])
        latex = _build_latex_table(results)
        assert '0.8123' in latex
