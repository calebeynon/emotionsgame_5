"""
Build round-level DiD panel dataset for sucker analysis (Issue #20).

Derives per-round suckering events from behavior and promise data, computes
event-study variables (tau, post, did_sample), and merges sentiment. A player
is "suckered" in a round if they contributed 25 and a groupmate who made a
promise contributed below the threshold.

Author: Claude Code
Date: 2026-02-05
"""

from pathlib import Path

import numpy as np
import pandas as pd

# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / 'datastore' / 'derived'
BEHAVIOR_FILE = DERIVED_DIR / 'behavior_classifications.csv'
PROMISE_FILE = DERIVED_DIR / 'promise_classifications.csv'
REGRESSION_FILE = DERIVED_DIR / 'issue_17_regression_data.csv'
OUTPUT_FILE = DERIVED_DIR / 'issue_20_did_panel.csv'

# MERGE KEYS
PLAYER_ROUND_KEYS = ['session_code', 'segment', 'round', 'label']
GROUP_ROUND_KEYS = ['session_code', 'segment', 'round', 'group']
PLAYER_SEGMENT_KEYS = ['session_code', 'segment', 'label']

# THRESHOLDS: groupmate must contribute below this to count as breaking promise
THRESHOLDS = {'20': 20, '5': 5}

# SENTIMENT COLUMN to merge from regression data
SENTIMENT_COL = 'sentiment_compound_mean'

EXPECTED_ROWS = 3520


# =====
# Main function
# =====
def main():
    """Main execution flow."""
    behavior_df = pd.read_csv(BEHAVIOR_FILE)
    promise_df = pd.read_csv(PROMISE_FILE)
    regression_df = pd.read_csv(REGRESSION_FILE)

    print(f"Loaded behavior: {len(behavior_df):,} rows")
    print(f"Loaded promises: {len(promise_df):,} rows")
    print(f"Loaded regression: {len(regression_df):,} rows")

    panel = build_suckered_flags(behavior_df, promise_df)
    panel = add_did_variables(panel)
    panel = merge_sentiment(panel, regression_df)
    panel = add_cluster_id(panel)

    validate_output(panel)
    save_and_summarize(panel)


# =====
# Suckering detection
# =====
def build_suckered_flags(behavior_df, promise_df):
    """Add per-round suckered_this_round flags for each threshold."""
    # Get promise_count per player-round (only rounds 2+)
    promise_counts = promise_df[PLAYER_ROUND_KEYS + ['promise_count']].copy()
    promise_counts = promise_counts.rename(columns={'promise_count': 'gm_promise_count'})

    # Merge promise info onto behavior for groupmate lookups
    enriched = behavior_df.merge(
        promise_counts, on=PLAYER_ROUND_KEYS, how='left'
    )
    enriched['gm_promise_count'] = enriched['gm_promise_count'].fillna(0)

    for suffix, threshold in THRESHOLDS.items():
        col = f'suckered_this_round_{suffix}'
        behavior_df[col] = _compute_suckered_column(
            behavior_df, enriched, threshold
        )

    return behavior_df


def _compute_suckered_column(behavior_df, enriched, threshold):
    """Compute boolean suckered flag for a given threshold."""
    enriched = _flag_promise_breakers(enriched, threshold)
    return _match_suckered_to_candidates(behavior_df, enriched)


def _flag_promise_breakers(enriched, threshold):
    """Flag players who made a promise and contributed below threshold."""
    enriched = enriched.copy()
    enriched['is_breaker'] = (
        (enriched['gm_promise_count'] > 0) &
        (enriched['contribution'] < threshold)
    )
    return enriched


def _match_suckered_to_candidates(behavior_df, enriched):
    """Match breaker flags to candidate players (contributed 25, round > 1)."""
    results = pd.Series(False, index=behavior_df.index)
    candidate_mask = (
        (behavior_df['round'] > 1) & (behavior_df['contribution'] == 25)
    )
    candidates = behavior_df[candidate_mask]
    if candidates.empty:
        return results

    any_breaker = _find_groupmate_breakers(candidates, enriched)
    return _map_breakers_to_index(results, candidates, any_breaker)


def _find_groupmate_breakers(candidates, enriched):
    """For each candidate, check if any OTHER group member is a breaker."""
    gm_cols = GROUP_ROUND_KEYS + ['label', 'is_breaker']
    merged = candidates[GROUP_ROUND_KEYS + ['label']].merge(
        enriched[gm_cols], on=GROUP_ROUND_KEYS, suffixes=('', '_gm')
    )
    merged = merged[merged['label'] != merged['label_gm']]
    return (
        merged.groupby(GROUP_ROUND_KEYS + ['label'])['is_breaker']
        .any()
        .reset_index()
        .rename(columns={'is_breaker': 'suckered'})
    )


def _map_breakers_to_index(results, candidates, any_breaker):
    """Map groupmate-breaker results back to the original DataFrame index."""
    keyed = candidates[GROUP_ROUND_KEYS + ['label']].reset_index()
    keyed = keyed.merge(
        any_breaker, on=GROUP_ROUND_KEYS + ['label'], how='left'
    )
    keyed['suckered'] = keyed['suckered'].fillna(False)
    results.loc[keyed['index']] = keyed['suckered'].values
    return results


# =====
# DiD event-study variables
# =====
def add_did_variables(df):
    """Add event count, first_suckered_round, tau, post, did_sample."""
    for suffix in THRESHOLDS:
        df = _add_did_vars_for_threshold(df, suffix)
    return df


def _add_did_vars_for_threshold(df, suffix):
    """Compute DiD variables for one threshold."""
    suckered_col = f'suckered_this_round_{suffix}'
    df = _merge_event_counts(df, suckered_col, suffix)
    df = _merge_first_suckered_round(df, suckered_col, suffix)
    df = _compute_tau_post_sample(df, suffix)
    return df


def _merge_event_counts(df, suckered_col, suffix):
    """Count suckering events per player-segment and merge back."""
    event_counts = (
        df[df[suckered_col]]
        .groupby(PLAYER_SEGMENT_KEYS)
        .size()
        .reset_index(name=f'suckered_event_count_{suffix}')
    )
    df = df.merge(event_counts, on=PLAYER_SEGMENT_KEYS, how='left')
    df[f'suckered_event_count_{suffix}'] = (
        df[f'suckered_event_count_{suffix}'].fillna(0).astype(int)
    )
    return df


def _merge_first_suckered_round(df, suckered_col, suffix):
    """Find earliest suckered round per player-segment and merge back."""
    first_round = (
        df[df[suckered_col]]
        .groupby(PLAYER_SEGMENT_KEYS)['round']
        .min()
        .reset_index(name=f'first_suckered_round_{suffix}')
    )
    return df.merge(first_round, on=PLAYER_SEGMENT_KEYS, how='left')


def _compute_tau_post_sample(df, suffix):
    """Compute tau, post, got_suckered, and did_sample columns."""
    fsr = f'first_suckered_round_{suffix}'
    has_event = df[fsr].notna()

    df[f'got_suckered_{suffix}'] = has_event
    df[f'tau_{suffix}'] = np.where(
        has_event, df['round'] - df[fsr], np.nan
    )
    df[f'post_{suffix}'] = np.where(
        has_event, (df[f'tau_{suffix}'] >= 0).astype(float), np.nan
    )
    df[f'did_sample_{suffix}'] = (
        (df[f'suckered_event_count_{suffix}'] == 1) | (~has_event)
    )
    return df


# =====
# Sentiment merge and cluster ID
# =====
def merge_sentiment(panel, regression_df):
    """Merge sentiment_compound_mean from regression data."""
    sentiment = regression_df[PLAYER_ROUND_KEYS + [SENTIMENT_COL]].copy()
    panel = panel.merge(sentiment, on=PLAYER_ROUND_KEYS, how='left')
    return panel


def add_cluster_id(df):
    """Add cluster_id = '{session_code}_{segment}_{group}'."""
    df['cluster_id'] = (
        df['session_code'] + '_' + df['segment'] + '_' + df['group'].astype(str)
    )
    return df


# =====
# Validation and output
# =====
def validate_output(df):
    """Validate output has expected row count and structure."""
    assert len(df) == EXPECTED_ROWS, (
        f"Expected {EXPECTED_ROWS} rows, got {len(df)}"
    )

    # Check no duplicate rows
    dupes = df.duplicated(subset=PLAYER_ROUND_KEYS).sum()
    assert dupes == 0, f"Found {dupes} duplicate player-round rows"

    # Validate known example: player N, sa7mprty, supergame4, round 4
    _validate_known_example(df)
    print("\nValidation passed!")


def _validate_known_example(df):
    """Verify player N, sa7mprty, supergame4 suckering in round 4."""
    player_n = _get_known_example_rows(df)
    assert player_n[player_n['round'] == 4]['suckered_this_round_20'].iloc[0], (
        "Player N should be suckered_this_round_20 in round 4"
    )
    assert not player_n[player_n['round'] == 3]['suckered_this_round_20'].iloc[0], (
        "Player N should NOT be suckered in round 3"
    )
    fsr = player_n[player_n['round'] == 4]['first_suckered_round_20'].iloc[0]
    assert fsr == 4, f"first_suckered_round_20 should be 4, got {fsr}"
    print("  Known example (player N, sa7mprty, supergame4) validated")


def _get_known_example_rows(df):
    """Extract player N, sa7mprty, supergame4 rows sorted by round."""
    mask = (
        (df['session_code'] == 'sa7mprty') &
        (df['segment'] == 'supergame4') &
        (df['label'] == 'N')
    )
    return df[mask].sort_values('round')


def save_and_summarize(df):
    """Save output CSV and print summary statistics."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Shape: {df.shape}")

    _print_suckering_summary(df)
    _print_did_sample_summary(df)
    _print_tau_distribution(df)


def _print_suckering_summary(df):
    """Print suckering event distribution."""
    print("\n--- Suckered event count distribution (threshold 20) ---")
    counts = df.groupby(PLAYER_SEGMENT_KEYS).first()
    dist = counts['suckered_event_count_20'].value_counts().sort_index()
    for val, n in dist.items():
        print(f"  {val} events: {n} player-segments")


def _print_did_sample_summary(df):
    """Print DiD sample inclusion counts."""
    for suffix in THRESHOLDS:
        unique_ps = df.groupby(PLAYER_SEGMENT_KEYS).first()
        n_sample = unique_ps[f'did_sample_{suffix}'].sum()
        n_total = len(unique_ps)
        print(f"\nDiD sample ({suffix}): {n_sample} / {n_total} player-segments")
        print(f"  got_suckered: {unique_ps[f'got_suckered_{suffix}'].sum()}")


def _print_tau_distribution(df):
    """Print tau distribution for treated players in DiD sample."""
    print("\n--- Tau distribution (threshold 20, DiD sample, treated) ---")
    mask = df['did_sample_20'] & df['got_suckered_20']
    tau_dist = df[mask]['tau_20'].value_counts().sort_index()
    for tau, n in tau_dist.items():
        print(f"  tau={tau:+.0f}: {n} obs")


# %%
if __name__ == "__main__":
    main()
