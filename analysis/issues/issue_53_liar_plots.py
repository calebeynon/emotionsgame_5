"""
Box plots of sentiment and facial emotion by liar bucket.
Author: Claude Code | Date: 2026-04-09
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# FILE PATHS
DATA_DIR = Path(__file__).parent.parent / 'datastore' / 'derived'
MERGED_PANEL = DATA_DIR / 'merged_panel.csv'
LIAR_BUCKETS = DATA_DIR / 'liar_buckets.csv'
OUTPUT_PLOTS = Path(__file__).parent.parent / 'output' / 'plots'
OUTPUT_TABLES = Path(__file__).parent.parent / 'output' / 'summary_statistics'

# CONSTANTS
BUCKET_ORDER = ['never', 'one_time', 'moderate', 'severe']
BUCKET_DISPLAY = ['Never', 'One-Time', 'Moderate', 'Severe']
EMOTION_COLS = [
    'emotion_anger', 'emotion_contempt', 'emotion_joy',
    'emotion_sadness', 'emotion_surprise',
]
EMOTION_LABELS = ['Anger', 'Contempt', 'Joy', 'Sadness', 'Surprise']


# =====
# Main function
# =====
def main():
    """Main execution flow for liar bucket visualizations."""
    df = load_and_merge()
    df = filter_results_pages(df)
    plot_sentiment_boxplot(df)
    plot_emotion_boxplots(df)
    save_summary_table(df)
    print("All outputs generated.")


# =====
# Data loading and merging
# =====
def load_and_merge() -> pd.DataFrame:
    """Load merged panel and liar buckets, merge on session_code + label."""
    validate_inputs_exist()
    panel = pd.read_csv(MERGED_PANEL)
    buckets = pd.read_csv(LIAR_BUCKETS, usecols=['session_code', 'label', 'liar_bucket'])
    merged = panel.merge(buckets, on=['session_code', 'label'], how='left')
    merged['liar_bucket'] = pd.Categorical(
        merged['liar_bucket'], categories=BUCKET_ORDER, ordered=True
    )
    return merged


def validate_inputs_exist():
    """Check that input files exist before proceeding."""
    if not MERGED_PANEL.exists():
        raise FileNotFoundError(f"Missing: {MERGED_PANEL}. Run merge_panel_data.py first.")
    if not LIAR_BUCKETS.exists():
        raise FileNotFoundError(f"Missing: {LIAR_BUCKETS}. Run liar_buckets.py first.")


def filter_results_pages(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to Results pages, drop round 1 (no sentiment data)."""
    results = df[df['page_type'] == 'Results'].copy()
    results = results[results['round'] > 1]
    print(f"Filtered to {len(results)} Results rows (round > 1)")
    return results


# =====
# Sentiment box plot
# =====
def plot_sentiment_boxplot(df: pd.DataFrame):
    """Box plot of sentiment_compound_mean by liar bucket."""
    plot_df = df.dropna(subset=['sentiment_compound_mean'])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=plot_df, x='liar_bucket', y='sentiment_compound_mean',
        hue='liar_bucket', order=BUCKET_ORDER, ax=ax, palette='Set2',
        legend=False,
    )
    add_n_labels(ax, plot_df, 'liar_bucket', BUCKET_ORDER)
    ax.set_xticks(range(len(BUCKET_DISPLAY)))
    ax.set_xticklabels(BUCKET_DISPLAY)
    ax.set_xlabel('Liar Bucket')
    ax.set_ylabel('Sentiment Compound Mean')
    ax.set_title('Chat Sentiment by Liar Bucket')
    fig.tight_layout()
    outpath = OUTPUT_PLOTS / 'sentiment_by_liar_bucket.png'
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved: {outpath}")


def add_n_labels(ax, df: pd.DataFrame, group_col: str, order: list):
    """Add sample size labels above each box plot category."""
    for i, cat in enumerate(order):
        n = df[df[group_col] == cat][group_col].count()
        ax.text(i, ax.get_ylim()[1], f'n={n}', ha='center', va='bottom', fontsize=8)


# =====
# Emotion faceted box plots
# =====
def plot_emotion_boxplots(df: pd.DataFrame):
    """Faceted box plots of key emotions by liar bucket."""
    long_df = melt_emotions(df)
    fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=False)
    for i, (emotion_label, ax) in enumerate(zip(EMOTION_LABELS, axes)):
        subset = long_df[long_df['Emotion'] == emotion_label]
        _draw_emotion_facet(ax, subset, emotion_label, i)
    fig.suptitle('Facial Emotions by Liar Bucket', fontsize=14, y=1.02)
    fig.tight_layout()
    outpath = OUTPUT_PLOTS / 'emotions_by_liar_bucket.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outpath}")


def _draw_emotion_facet(ax, subset: pd.DataFrame, label: str, idx: int):
    """Draw a single emotion facet with box plot and formatted axis."""
    sns.boxplot(
        data=subset, x='liar_bucket', y='value',
        hue='liar_bucket', order=BUCKET_ORDER, ax=ax, palette='Set2',
        legend=False,
    )
    ax.set_title(label)
    ax.set_xlabel('Liar Bucket' if idx == 2 else '')
    ax.set_ylabel('Intensity' if idx == 0 else '')
    ax.set_xticks(range(len(BUCKET_DISPLAY)))
    ax.set_xticklabels(BUCKET_DISPLAY, rotation=45, ha='right')


def melt_emotions(df: pd.DataFrame) -> pd.DataFrame:
    """Melt emotion columns into long format for faceted plotting."""
    col_map = dict(zip(EMOTION_COLS, EMOTION_LABELS))
    melted = df.melt(
        id_vars=['session_code', 'label', 'liar_bucket'],
        value_vars=EMOTION_COLS,
        var_name='Emotion', value_name='value',
    )
    melted['Emotion'] = melted['Emotion'].map(col_map)
    return melted.dropna(subset=['value'])


# =====
# Summary statistics table
# =====
def save_summary_table(df: pd.DataFrame):
    """Save LaTeX table with means and SDs by liar bucket."""
    all_cols = ['sentiment_compound_mean'] + EMOTION_COLS
    all_labels = ['Sentiment'] + EMOTION_LABELS
    rows = build_summary_rows(df, all_cols, all_labels)
    latex = format_latex_table(rows)
    outpath = OUTPUT_TABLES / 'liar_bucket_summary.tex'
    outpath.write_text(latex)
    print(f"Saved: {outpath}")


def build_summary_rows(df: pd.DataFrame, cols: list, labels: list) -> list:
    """Build summary stat rows: mean (SD) per bucket for each variable."""
    rows = []
    for col, label in zip(cols, labels):
        row = [label]
        for bucket in BUCKET_ORDER:
            subset = df[df['liar_bucket'] == bucket][col].dropna()
            row.append(format_mean_sd(subset))
        rows.append(row)
    return rows


def format_mean_sd(series: pd.Series) -> str:
    """Format as 'mean (SD)' string for LaTeX."""
    if len(series) == 0:
        return '---'
    return f'{series.mean():.3f} ({series.std():.3f})'


def format_latex_table(rows: list) -> str:
    """Format rows into a LaTeX tabular environment."""
    header = (
        '\\begin{tabular}{lcccc}\n'
        '\\toprule\n'
        ' & Never & One-Time & Moderate & Severe \\\\\n'
        '\\midrule\n'
    )
    body = ''.join(f'{" & ".join(r)} \\\\\n' for r in rows)
    footer = '\\bottomrule\n\\end{tabular}\n'
    return header + body + footer


# %%
if __name__ == "__main__":
    main()
