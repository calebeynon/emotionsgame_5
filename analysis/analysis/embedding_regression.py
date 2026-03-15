"""
Logistic regression baselines comparing VADER, embedding projections, and full embeddings.

Runs 5-fold cross-validated logistic regression on group-round level data,
comparing four feature sets for predicting cooperative vs non-cooperative state.
Outputs a LaTeX comparison table.

Author: Claude Code
Date: 2026-03-15
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Add derived directory for analyze_embeddings imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'derived'))

# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / 'datastore' / 'derived'
EMBEDDINGS_SMALL = DERIVED_DIR / 'embeddings_small.parquet'
PROJECTIONS_FILE = DERIVED_DIR / 'embedding_projections.csv'
SENTIMENT_FILE = DERIVED_DIR / 'sentiment_scores.csv'
STATE_FILE = DERIVED_DIR / 'player_state_classification.csv'
OUTPUT_DIR = Path(__file__).parent.parent / 'output' / 'tables'
OUTPUT_FILE = OUTPUT_DIR / 'embedding_regression_comparison.tex'

# MODEL COMPARISON CONFIG
N_FOLDS = 5
PCA_COMPONENTS = 50
RANDOM_STATE = 42
GROUP_KEYS = ['session_code', 'segment', 'round', 'group']

VADER_FEATURES = [
    'sentiment_compound_mean', 'sentiment_positive_mean',
    'sentiment_negative_mean', 'sentiment_neutral_mean',
]


# =====
# Main function
# =====
def main():
    """Main execution flow."""
    dataset = build_dataset(
        EMBEDDINGS_SMALL, SENTIMENT_FILE, STATE_FILE, PROJECTIONS_FILE,
    )
    print(f"Dataset: {len(dataset)} group-rounds")

    results = run_model_comparison(dataset)
    print_summary(results)
    save_comparison_table(results, OUTPUT_FILE)


# =====
# Dataset construction
# =====
def build_dataset(
    embeddings_path: Path,
    sentiment_path: Path,
    state_path: Path,
    projections_path: Path,
) -> pd.DataFrame:
    """Merge group-round embeddings, VADER sentiment, and state labels."""
    emb_df = _load_group_embeddings(embeddings_path)
    vader_df = _aggregate_sentiment(sentiment_path)
    state_df = _aggregate_state(state_path)
    proj_df = _aggregate_projections(projections_path)

    merged = emb_df.merge(vader_df, on=GROUP_KEYS, how='inner')
    merged = merged.merge(state_df, on=GROUP_KEYS, how='inner')
    merged = merged.merge(proj_df, on=GROUP_KEYS, how='inner')
    return merged


def _load_group_embeddings(path: Path) -> pd.DataFrame:
    """Load embeddings and aggregate to group-round means."""
    df = pd.read_parquet(path)
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    return df.groupby(GROUP_KEYS)[emb_cols].mean().reset_index()


def _aggregate_sentiment(path: Path) -> pd.DataFrame:
    """Aggregate player-level VADER scores to group-round means."""
    df = pd.read_csv(path)
    return df.groupby(GROUP_KEYS)[VADER_FEATURES].mean().reset_index()


def _aggregate_state(path: Path) -> pd.DataFrame:
    """Aggregate player states to group-round majority vote."""
    df = pd.read_csv(path)
    df = df.rename(columns={'round_num': 'round', 'group_id': 'group'})

    grouped = df.groupby(GROUP_KEYS)['player_state'].apply(
        _majority_vote
    ).reset_index()
    grouped['cooperative'] = (grouped['player_state'] == 'cooperative').astype(int)
    return grouped[GROUP_KEYS + ['cooperative']]


def _aggregate_projections(path: Path) -> pd.DataFrame:
    """Aggregate message-level projections to group-round means."""
    df = pd.read_csv(path)
    proj_cols = [c for c in df.columns if c.startswith('projection_score')]
    return df.groupby(GROUP_KEYS)[proj_cols].mean().reset_index()


def _majority_vote(states: pd.Series) -> str:
    """Return most common state label. Ties go to cooperative."""
    counts = states.value_counts()
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        return 'cooperative'
    return counts.index[0]


# =====
# Model comparison
# =====
def run_model_comparison(dataset: pd.DataFrame) -> pd.DataFrame:
    """Run 5-fold CV for 4 model configurations, return metrics."""
    models = _define_models(dataset)
    y = dataset['cooperative'].values

    records = []
    for name, X in models.items():
        use_pca = (name == 'Full embedding (PCA 50)')
        metrics = cross_validate_model(X, y, use_pca=use_pca)
        metrics['model'] = name
        records.append(metrics)

    return pd.DataFrame(records)


def _define_models(dataset: pd.DataFrame) -> dict:
    """Define feature matrices for each model configuration."""
    emb_cols = [c for c in dataset.columns if c.startswith('emb_')]
    proj_cols = [c for c in dataset.columns if c.startswith('projection_score')]
    return {
        'VADER only': dataset[VADER_FEATURES].values,
        'Projection only': dataset[proj_cols].values,
        'Full embedding (PCA 50)': dataset[emb_cols].values,
        'VADER + Projection': dataset[VADER_FEATURES + proj_cols].values,
    }


def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    use_pca: bool = False,
) -> dict:
    """Run stratified 5-fold CV and return mean metrics."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []

    for train_idx, test_idx in skf.split(X, y):
        metrics = _evaluate_fold(X, y, train_idx, test_idx, use_pca)
        fold_metrics.append(metrics)

    return _average_metrics(fold_metrics)


def _evaluate_fold(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    use_pca: bool,
) -> dict:
    """Train and evaluate one fold, return metric dict."""
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train, X_test = _preprocess(X_train, X_test, use_pca)

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    return _compute_metrics(clf, X_test, y_test)


def _preprocess(
    X_train: np.ndarray, X_test: np.ndarray, use_pca: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Scale features and optionally apply PCA."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if use_pca:
        n_components = min(PCA_COMPONENTS, X_train.shape[1], X_train.shape[0])
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    return X_train, X_test


def _compute_metrics(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute classification metrics for one fold."""
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }


def _average_metrics(fold_metrics: list[dict]) -> dict:
    """Average metrics across folds."""
    keys = ['accuracy', 'auc', 'precision', 'recall', 'f1']
    return {k: np.mean([m[k] for m in fold_metrics]) for k in keys}


# =====
# Output
# =====
def print_summary(results: pd.DataFrame) -> None:
    """Print comparison table to stdout."""
    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION MODEL COMPARISON (5-fold CV)")
    print("=" * 70)
    metrics = ['accuracy', 'auc', 'precision', 'recall', 'f1']
    print(f"{'Model':<25} " + " ".join(f"{m:>10}" for m in metrics))
    print("-" * 70)

    for _, row in results.iterrows():
        vals = " ".join(f"{row[m]:>10.4f}" for m in metrics)
        print(f"{row['model']:<25} {vals}")
    print("=" * 70)


def save_comparison_table(results: pd.DataFrame, output_path: Path) -> None:
    """Save results as a LaTeX table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latex = _build_latex_table(results)
    output_path.write_text(latex)
    print(f"\nLaTeX table saved to {output_path}")


def _build_latex_table(results: pd.DataFrame) -> str:
    """Build LaTeX tabular string from results DataFrame."""
    metrics = ['accuracy', 'auc', 'precision', 'recall', 'f1']
    header = " & ".join(["Model"] + [m.upper() for m in metrics])
    lines = _latex_header(header, len(metrics))

    for _, row in results.iterrows():
        vals = " & ".join(f"{row[m]:.4f}" for m in metrics)
        lines.append(f"   {row['model']} & {vals} \\\\")

    lines += _latex_footer()
    return "\n".join(lines) + "\n"


def _latex_header(header: str, n_metrics: int) -> list[str]:
    """Build LaTeX table header lines."""
    return [
        r"\begingroup", r"\centering",
        r"\begin{tabular}{l" + "c" * n_metrics + "}",
        r"   \tabularnewline \midrule \midrule",
        f"   {header} \\\\", r"   \midrule",
    ]


def _latex_footer() -> list[str]:
    """Build LaTeX table footer lines."""
    return [
        r"   \midrule \midrule",
        r"   \multicolumn{6}{l}{\emph{5-fold stratified cross-validation}}\\",
        r"\end{tabular}", r"\par\endgroup",
    ]


# %%
if __name__ == "__main__":
    main()
