"""
Build round-level DiD panel for sucker/liar analysis (Issue #20).
Author: Claude Code  Date: 2026-02-05
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
PLAYER_KEYS = ['session_code', 'label']

THRESHOLDS = {'20': 20, '5': 5}
SENTIMENT_COL = 'sentiment_compound_mean'
EXPECTED_ROWS = 3520
SEGMENT_ORDER = {f'supergame{i}': i for i in range(1, 6)}

COL_TEMPLATES = {
    ('suckered', 'event_count'): 'suckered_event_count_{s}',
    ('suckered', 'first_round'): 'first_suckered_round_{s}',
    ('suckered', 'tau'): 'tau_{s}',
    ('suckered', 'treated'): 'got_suckered_{s}',
    ('suckered', 'post'): 'post_{s}',
    ('suckered', 'did_sample'): 'did_sample_{s}',
    ('liar', 'event_count'): 'liar_event_count_{s}',
    ('liar', 'first_round'): 'first_lied_round_{s}',
    ('liar', 'tau'): 'liar_tau_{s}',
    ('liar', 'treated'): 'is_liar_did_{s}',
    ('liar', 'post'): 'liar_post_{s}',
    ('liar', 'did_sample'): 'liar_did_sample_{s}',
}

EVENT_SPECS = [
    ('suckered_this_round_{s}', 'suckered'),
    ('lied_this_round_{s}', 'liar'),
]


def main():
    """Main execution flow."""
    behavior_df = pd.read_csv(BEHAVIOR_FILE)
    promise_df = pd.read_csv(PROMISE_FILE)
    regression_df = pd.read_csv(REGRESSION_FILE)
    print(f"Loaded: {len(behavior_df):,} behavior, {len(promise_df):,} promise, {len(regression_df):,} regression")

    panel = build_suckered_flags(behavior_df, promise_df)
    panel = build_liar_flags(panel, promise_df)
    panel = add_did_variables(panel)
    panel = add_always_cooperator_flags(panel)
    panel = add_cross_segment_spillover(panel)
    sentiment = regression_df[PLAYER_ROUND_KEYS + [SENTIMENT_COL]].copy()
    panel = panel.merge(sentiment, on=PLAYER_ROUND_KEYS, how='left')
    panel['cluster_id'] = (
        panel['session_code'] + '_' + panel['segment'] + '_' + panel['group'].astype(str)
    )
    validate_output(panel)
    save_and_summarize(panel)


def build_suckered_flags(behavior_df, promise_df):
    """Add per-round suckered_this_round flags for each threshold."""
    pcounts = promise_df[PLAYER_ROUND_KEYS + ['promise_count']].copy()
    pcounts = pcounts.rename(columns={'promise_count': 'gm_promise_count'})
    enriched = behavior_df.merge(pcounts, on=PLAYER_ROUND_KEYS, how='left')
    enriched['gm_promise_count'] = enriched['gm_promise_count'].fillna(0)

    for suffix, threshold in THRESHOLDS.items():
        behavior_df[f'suckered_this_round_{suffix}'] = _compute_suckered(
            behavior_df, enriched, threshold
        )
    return behavior_df


def _compute_suckered(behavior_df, enriched, threshold):
    """Compute boolean suckered flag for a given threshold."""
    enriched = enriched.copy()
    enriched['is_breaker'] = (
        (enriched['gm_promise_count'] > 0) & (enriched['contribution'] < threshold)
    )
    return _match_suckered_to_candidates(behavior_df, enriched)


def _match_suckered_to_candidates(behavior_df, enriched):
    """Match breaker flags to candidate players (contributed 25, round > 1)."""
    results = pd.Series(False, index=behavior_df.index)
    mask = (behavior_df['round'] > 1) & (behavior_df['contribution'] == 25)
    candidates = behavior_df[mask]
    if candidates.empty:
        return results
    any_breaker = _find_groupmate_breakers(candidates, enriched)
    keyed = candidates[GROUP_ROUND_KEYS + ['label']].reset_index()
    keyed = keyed.merge(any_breaker, on=GROUP_ROUND_KEYS + ['label'], how='left')
    keyed['suckered'] = keyed['suckered'].fillna(False)
    results.loc[keyed['index']] = keyed['suckered'].values
    return results


def _find_groupmate_breakers(candidates, enriched):
    """For each candidate, check if any OTHER group member is a breaker."""
    merged = candidates[GROUP_ROUND_KEYS + ['label']].merge(
        enriched[GROUP_ROUND_KEYS + ['label', 'is_breaker']],
        on=GROUP_ROUND_KEYS, suffixes=('', '_gm'),
    )
    merged = merged[merged['label'] != merged['label_gm']]
    return (
        merged.groupby(GROUP_ROUND_KEYS + ['label'])['is_breaker']
        .any().reset_index().rename(columns={'is_breaker': 'suckered'})
    )


def build_liar_flags(df, promise_df):
    """Add per-round lied_this_round flags for each threshold."""
    pcounts = promise_df[PLAYER_ROUND_KEYS + ['promise_count']].copy()
    merged = df.merge(pcounts, on=PLAYER_ROUND_KEYS, how='left')
    merged['promise_count'] = merged['promise_count'].fillna(0)
    for suffix, threshold in THRESHOLDS.items():
        df[f'lied_this_round_{suffix}'] = (
            (merged['round'] > 1)
            & (merged['promise_count'] > 0)
            & (merged['contribution'] < threshold)
        )
    return df


def add_did_variables(df):
    """Add event-study DiD variables for sucker and (if present) liar events."""
    for suffix in THRESHOLDS:
        for event_tpl, prefix in EVENT_SPECS:
            event_col = event_tpl.format(s=suffix)
            if event_col in df.columns:
                df = _add_event_study_vars(df, event_col, prefix, suffix)
    return df


def _col(prefix, var, suffix):
    """Resolve column name for a prefix/var/suffix combination."""
    return COL_TEMPLATES[(prefix, var)].format(s=suffix)


def _add_event_study_vars(df, event_col, prefix, suffix):
    """Compute event count, first round, tau, treated flag, did_sample."""
    events = df[df[event_col]]
    cnt, fst = _col(prefix, 'event_count', suffix), _col(prefix, 'first_round', suffix)
    df = _merge_count_and_first(df, events, cnt, fst)
    has_event = df[fst].notna()
    df[_col(prefix, 'treated', suffix)] = has_event
    df[_col(prefix, 'tau', suffix)] = np.where(has_event, df['round'] - df[fst], np.nan)
    df[_col(prefix, 'post', suffix)] = np.where(
        has_event, (df['round'] >= df[fst]).astype(float), np.nan
    )
    df[_col(prefix, 'did_sample', suffix)] = (df[cnt] == 1) | (~has_event)
    return df


def _merge_count_and_first(df, events, cnt_col, fst_col):
    """Merge event count and first event round onto df."""
    counts = events.groupby(PLAYER_SEGMENT_KEYS).size().reset_index(name=cnt_col)
    df = df.merge(counts, on=PLAYER_SEGMENT_KEYS, how='left')
    df[cnt_col] = df[cnt_col].fillna(0).astype(int)
    first = events.groupby(PLAYER_SEGMENT_KEYS)['round'].min().reset_index(name=fst_col)
    return df.merge(first, on=PLAYER_SEGMENT_KEYS, how='left')


def add_always_cooperator_flags(df):
    """Add always-cooperator flags and robustness DiD sample indicators."""
    full_coop = _full_cooperator_mask(df)
    for suffix in THRESHOLDS:
        ac = f'always_coop_{suffix}'
        df[ac] = full_coop & ~df[f'got_suckered_{suffix}']
        cnt = _col('suckered', 'event_count', suffix)
        df[f'did_sample_robust_{suffix}'] = (df[cnt] == 1) | df[ac]
    return df


def _full_cooperator_mask(df):
    """True for players who contributed 25 in every round > 1 of the segment."""
    eligible = df[df['round'] > 1].copy()
    missed = eligible[eligible['contribution'] != 25]
    defectors = set(missed[PLAYER_SEGMENT_KEYS].apply(tuple, axis=1))
    keys = df[PLAYER_SEGMENT_KEYS].apply(tuple, axis=1)
    return ~keys.isin(defectors)


def add_cross_segment_spillover(df):
    """Add flags for whether player was suckered in a prior segment."""
    df['segment_number'] = df['segment'].map(SEGMENT_ORDER)
    for suffix in THRESHOLDS:
        df = _add_spillover_for_threshold(df, suffix)
    df.drop(columns=['segment_number'], inplace=True)
    return df


def _add_spillover_for_threshold(df, suffix):
    """Compute cross-segment spillover columns for one threshold."""
    fss = f'first_suckered_segment_{suffix}'
    suckered_ps = df[df[f'got_suckered_{suffix}']].copy()
    if suckered_ps.empty:
        df[fss] = np.nan
        df[f'suckered_prior_segment_{suffix}'] = False
        df[f'segments_since_suckered_{suffix}'] = np.nan
        return df
    return _compute_spillover_cols(df, suckered_ps, fss, suffix)


def _compute_spillover_cols(df, suckered_ps, fss, suffix):
    """Compute spillover from first-suckered segment data."""
    first_seg = (
        suckered_ps.groupby(PLAYER_KEYS)['segment_number'].min()
        .reset_index(name=fss)
    )
    df = df.merge(first_seg, on=PLAYER_KEYS, how='left')
    df[f'suckered_prior_segment_{suffix}'] = (df['segment_number'] > df[fss]).fillna(False)
    diff = df['segment_number'] - df[fss]
    df[f'segments_since_suckered_{suffix}'] = np.where(diff >= 0, diff, np.nan)
    return df


def validate_output(df):
    """Validate output has expected row count and required columns."""
    assert len(df) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, got {len(df)}"
    dupes = df.duplicated(subset=PLAYER_ROUND_KEYS).sum()
    assert dupes == 0, f"Found {dupes} duplicate player-round rows"
    _check_required_columns(df)
    _validate_known_example(df)
    print("\nValidation passed!")


def _check_required_columns(df):
    """Check that all expected columns exist."""
    required = ['cluster_id', SENTIMENT_COL]
    for suffix in THRESHOLDS:
        for prefix in ('suckered', 'liar'):
            for var in ('event_count', 'first_round', 'tau', 'treated', 'post', 'did_sample'):
                required.append(_col(prefix, var, suffix))
        required += [
            f'suckered_this_round_{suffix}', f'lied_this_round_{suffix}',
            f'first_suckered_segment_{suffix}', f'suckered_prior_segment_{suffix}',
            f'segments_since_suckered_{suffix}',
            f'always_coop_{suffix}', f'did_sample_robust_{suffix}',
        ]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"
    print(f"  All {len(required)} required columns present")


def _validate_known_example(df):
    """Verify player N, sa7mprty, supergame4 suckering in round 4."""
    mask = (
        (df['session_code'] == 'sa7mprty')
        & (df['segment'] == 'supergame4') & (df['label'] == 'N')
    )
    pn = df[mask].sort_values('round')
    assert pn[pn['round'] == 4]['suckered_this_round_20'].iloc[0]
    assert not pn[pn['round'] == 3]['suckered_this_round_20'].iloc[0]
    fsr = pn[pn['round'] == 4]['first_suckered_round_20'].iloc[0]
    assert fsr == 4, f"first_suckered_round_20 should be 4, got {fsr}"
    print("  Known example (player N, sa7mprty, supergame4) validated")


def save_and_summarize(df):
    """Save output CSV and print summary statistics."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}\nShape: {df.shape}")
    u = df.groupby(PLAYER_SEGMENT_KEYS).first()
    n = len(u)
    for pfx, lbl in [('suckered', 'Suckered'), ('liar', 'Liar')]:
        for s in THRESHOLDS:
            print(f"  {lbl} DiD ({s}): {u[_col(pfx, 'did_sample', s)].sum()}/{n} sample,"
                  f" {u[_col(pfx, 'treated', s)].sum()} treated")
    for s in THRESHOLDS:
        print(f"  Robust DiD ({s}): {u[f'did_sample_robust_{s}'].sum()}/{n} sample,"
              f" {u[f'always_coop_{s}'].sum()} always-coop controls")
        print(f"  Suckered prior segment ({s}): {u[f'suckered_prior_segment_{s}'].sum()} player-segments")


# %%
if __name__ == "__main__":
    main()
