"""
Build lying-contagion panel for issue #72.

Derives lag-based predictors of within-group lying contagion from the
behavior classifications CSV: whether other group members lied in the
prior round, and any prior lying within the current segment.

Author: Claude Code
Date: 2026-04-19
"""

from pathlib import Path

import pandas as pd

# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / 'datastore' / 'derived'
INPUT_FILE = DERIVED_DIR / 'behavior_classifications.csv'
OUTPUT_FILE = DERIVED_DIR / 'issue_72_panel.csv'

# GROUPING KEYS
GROUP_KEYS = ['session_code', 'segment', 'group']
PLAYER_KEYS = ['session_code', 'segment', 'label']

OUTPUT_COLUMNS = [
    'session_code', 'treatment', 'segment', 'round', 'group', 'label',
    'lied', 'self_lied_lag', 'group_lied_lag',
    'any_self_lied_prior', 'any_group_lied_prior',
    'cluster_group', 'label_session',
]

EXPECTED_ROWS = 2720


# =====
# Main function (FIRST - shows high-level flow)
# =====
def main():
    """Main execution flow."""
    df = load_source()
    df = add_lied(df)
    df = add_self_lags(df)
    df = add_group_contagion(df)
    df = add_identifiers(df)
    df = drop_first_round(df)

    validate(df)

    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    df[OUTPUT_COLUMNS].to_csv(OUTPUT_FILE, index=False)
    print(f"\nOutput saved to: {OUTPUT_FILE}")
    print(f"Shape: {df.shape[0]:,} rows x {len(OUTPUT_COLUMNS)} columns")


# =====
# Load and normalize source data
# =====
def load_source() -> pd.DataFrame:
    """Read behavior_classifications.csv and sort deterministically."""
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded: {len(df):,} rows from {INPUT_FILE.name}")
    df = df.sort_values(GROUP_KEYS + ['round', 'label']).reset_index(drop=True)
    return df


def add_lied(df: pd.DataFrame) -> pd.DataFrame:
    """Cast lied_this_round_20 to 0/1 int via explicit dtype dispatch."""
    raw = df['lied_this_round_20']
    if raw.isna().any():
        raise ValueError(
            "lied_this_round_20 contains NaN; refusing to silently coerce. "
            "Fix the upstream behavior classifier to emit bool/0/1 only."
        )
    df['lied'] = _coerce_lied(raw)
    return df


# Accepted dtypes + string literals for lied_this_round_20.
STRING_TRUE = {'True', True}
STRING_FALSE = {'False', False}


def _coerce_lied(raw: pd.Series) -> pd.Series:
    """Dispatch bool/int/object dtypes to a clean int 0/1 series."""
    if raw.dtype == bool:
        return raw.astype(int)
    if pd.api.types.is_integer_dtype(raw):
        extra = set(raw.unique().tolist()) - {0, 1}
        if extra:
            raise ValueError(
                f"lied_this_round_20 integer values outside {{0,1}}: {sorted(extra)}"
            )
        return raw.astype(int)
    if pd.api.types.is_object_dtype(raw):
        return _coerce_object_lied(raw)
    raise ValueError(
        f"lied_this_round_20 has unsupported dtype {raw.dtype}; "
        f"expected bool, int, or string 'True'/'False'. "
        f"Values sample: {raw.head().tolist()}"
    )


def _coerce_object_lied(raw: pd.Series) -> pd.Series:
    """Map object-dtype 'True'/'False' (capitalized only) to int 0/1."""
    allowed = STRING_TRUE | STRING_FALSE
    extra = set(raw.unique().tolist()) - allowed
    if extra:
        raise ValueError(
            f"lied_this_round_20 object values outside whitelist "
            f"{{'True','False',True,False}}: {sorted(map(repr, extra))}. "
            "Lowercase 'true'/'false' are NOT accepted."
        )
    return raw.isin(STRING_TRUE).astype(int)


# =====
# Own-history lags (within session_code, segment, label)
# =====
def add_self_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add self_lied_lag and any_self_lied_prior within each player-segment."""
    df = df.sort_values(PLAYER_KEYS + ['round']).reset_index(drop=True)
    grouped = df.groupby(PLAYER_KEYS, sort=False)['lied']
    df['self_lied_lag'] = grouped.shift(1)
    df['any_self_lied_prior'] = grouped.cummax().shift(1)
    return df


# =====
# Group contagion (self-excluded lags within session_code, segment, group)
# =====
def add_group_contagion(df: pd.DataFrame) -> pd.DataFrame:
    """Add group_lied_lag and any_group_lied_prior using sum-minus-self (correct self-exclusion)."""
    df = df.sort_values(GROUP_KEYS + ['round', 'label']).reset_index(drop=True)
    # Sum-minus-self is required: max-minus-self gives 0 when BOTH self and a
    # groupmate lied in the same round. Sum-based arithmetic avoids that.
    group_sum = df.groupby(GROUP_KEYS + ['round'])['lied'].transform('sum')
    df['other_lied_this_round'] = ((group_sum - df['lied']) >= 1).astype(int)

    df = df.sort_values(PLAYER_KEYS + ['round']).reset_index(drop=True)
    pg = df.groupby(PLAYER_KEYS, sort=False)['other_lied_this_round']
    df['group_lied_lag'] = pg.shift(1)
    df['any_group_lied_prior'] = pg.cummax().shift(1)
    return df


# =====
# Identifier columns
# =====
def add_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Build cluster_group and label_session identifiers."""
    df['cluster_group'] = (
        df['session_code'].astype(str)
        + '_' + df['segment'].astype(str)
        + '_' + df['group'].astype(str)
    )
    df['label_session'] = df['label'].astype(str) + '_' + df['session_code'].astype(str)
    return df


# =====
# Drop first round per (session, segment, label) where self_lied_lag is NaN
# =====
def drop_first_round(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing self_lied_lag (first round per segment per player)."""
    before = len(df)
    df = df.dropna(subset=['self_lied_lag']).copy()
    print(f"Dropped {before - len(df):,} first-round rows")

    int_cols = [
        'self_lied_lag', 'group_lied_lag',
        'any_self_lied_prior', 'any_group_lied_prior',
    ]
    for col in int_cols:
        df[col] = df[col].astype(int)
    return df


# =====
# Validation
# =====
def validate(df: pd.DataFrame):
    """Fail loudly if row count or NaN expectations are violated."""
    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows, got {len(df):,}. "
            f"Check source CSV integrity or drop logic in drop_first_round()."
        )
    required = ['lied', 'group_lied_lag', 'self_lied_lag',
                'any_self_lied_prior', 'any_group_lied_prior']
    for col in required:
        n_nan = df[col].isna().sum()
        if n_nan:
            raise ValueError(f"Column {col} has {n_nan} NaN values after drop.")
    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required output columns: {missing}")
    print(f"Validation passed: {len(df):,} rows, no NaN in constructed vars")


# %%
if __name__ == "__main__":
    main()
