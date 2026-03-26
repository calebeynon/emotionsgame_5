"""
Logistic regression: can round-liar direction embeddings predict contributions?

Runs 5-fold cross-validated logistic regression at the player-round level,
comparing feature sets for predicting high contribution (>= 20/25).
Uses round-liar direction vector projections as features.

Author: Claude Code
Date: 2026-03-21
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

sys.path.insert(0, str(Path(__file__).parent.parent / 'derived'))

# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / 'datastore' / 'derived'
EMBEDDINGS_PR = DERIVED_DIR / 'embeddings_player_round_small.parquet'
RLIAR_PROJECTIONS_FILE = DERIVED_DIR / 'round_liar_embedding_projections.csv'
SENTIMENT_FILE = DERIVED_DIR / 'sentiment_scores.csv'
OUTPUT_DIR = Path(__file__).parent.parent / 'output' / 'tables'
OUTPUT_FILE = OUTPUT_DIR / 'round_liar_embedding_regression_comparison.tex'

# MODEL CONFIG
N_FOLDS = 5
PCA_COMPONENTS = 50
RANDOM_STATE = 42
PLAYER_KEYS = ['session_code', 'segment', 'round', 'group', 'label']
HIGH_CONTRIB_THRESHOLD = 20

VADER_FEATURES = [
    'sentiment_compound_mean', 'sentiment_positive_mean',
    'sentiment_negative_mean', 'sentiment_neutral_mean',
]


# =====
# Main function
# =====
def main():
    """Main execution flow."""
    dataset = build_dataset()
    print(f"Dataset: {len(dataset)} player-rounds")

    results = run_model_comparison(dataset)
    print_summary(results)
    save_comparison_table(results, OUTPUT_FILE)


# =====
# Dataset construction
# =====
def build_dataset() -> pd.DataFrame:
    """Merge player-round embeddings, VADER, projections, and contribution."""
    emb_df = _load_player_embeddings()
    vader_df = _load_sentiment()
    proj_df = _load_projections()

    merged = emb_df.merge(vader_df, on=PLAYER_KEYS, how='inner')
    merged = merged.merge(proj_df, on=PLAYER_KEYS, how='inner')
    merged = merged.dropna(subset=['high_contribution'])
    return merged


def _load_player_embeddings() -> pd.DataFrame:
    """Load player-round embeddings parquet."""
    df = pd.read_parquet(EMBEDDINGS_PR)
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    return df[PLAYER_KEYS + emb_cols]


def _load_sentiment() -> pd.DataFrame:
    """Load player-round VADER scores and contribution."""
    df = pd.read_csv(SENTIMENT_FILE)
    df['high_contribution'] = (df['contribution'] >= HIGH_CONTRIB_THRESHOLD).astype(int)
    return df[PLAYER_KEYS + VADER_FEATURES + ['high_contribution']]


def _load_projections() -> pd.DataFrame:
    """Aggregate message-level round-liar projections to player-round means."""
    df = pd.read_csv(RLIAR_PROJECTIONS_FILE)
    proj_cols = [c for c in df.columns if c.startswith('proj_rliar_')]
    return df.groupby(PLAYER_KEYS)[proj_cols].mean().reset_index()


# =====
# Model comparison
# =====
def run_model_comparison(dataset: pd.DataFrame) -> pd.DataFrame:
    """Run 5-fold CV for 4 model configurations."""
    models = _define_models(dataset)
    y = dataset['high_contribution'].values

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
    proj_cols = [c for c in dataset.columns if c.startswith('proj_rliar_')]
    return {
        'VADER only': dataset[VADER_FEATURES].values,
        'Round-liar proj. only': dataset[proj_cols].values,
        'Full embedding (PCA 50)': dataset[emb_cols].values,
        'VADER + Round-liar proj.': dataset[VADER_FEATURES + proj_cols].values,
    }


def cross_validate_model(
    X: np.ndarray, y: np.ndarray, use_pca: bool = False,
) -> dict:
    """Run stratified 5-fold CV and return mean metrics."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []

    for train_idx, test_idx in skf.split(X, y):
        metrics = _evaluate_fold(X, y, train_idx, test_idx, use_pca)
        fold_metrics.append(metrics)

    return _average_metrics(fold_metrics)


def _evaluate_fold(
    X: np.ndarray, y: np.ndarray,
    train_idx: np.ndarray, test_idx: np.ndarray, use_pca: bool,
) -> dict:
    """Train and evaluate one fold."""
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train, X_test = _preprocess(X_train, X_test, use_pca)

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    return _compute_metrics(clf, X_test, y_test)


def _preprocess(
    X_train: np.ndarray, X_test: np.ndarray, use_pca: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale features and optionally apply PCA."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if use_pca:
        n = min(PCA_COMPONENTS, X_train.shape[1], X_train.shape[0])
        pca = PCA(n_components=n, random_state=RANDOM_STATE)
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
    print("ROUND-LIAR PROJECTION -> PLAYER CONTRIBUTION (5-fold CV)")
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
        r"   \multicolumn{6}{l}{\emph{5-fold stratified CV, player-round level}}\\",
        r"\end{tabular}", r"\par\endgroup",
    ]


# %%
if __name__ == "__main__":
    main()
