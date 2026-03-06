"""
Purpose: End-to-end data pipeline verification for Issue #33 summary statistics.
         Traces individual player data from raw oTree exports through derived CSVs
         to final summary table values, ensuring no data corruption or loss.
Author: Claude
Date: 2026-03-02
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'analysis' / 'analysis' / 'summary_statistics'))

from ss_common import (
    DERIVED_DIR,
    PARTICIPATION_FEE,
    POINTS_TO_DOLLARS,
    RAW_DIR,
    SESSION_CODE_TO_TREATMENT,
    SUPERGAME_ROUNDS,
    load_contributions,
    load_payoffs,
    load_raw_data,
)

# TEST PLAYERS — one per session, spanning both treatments
# Session 03 has a code mismatch: raw uses z8dowljr, contributions.csv uses
# irrzlgk2, payoffs.csv uses z8dowljr. We track both codes explicitly.
TEST_PLAYERS = [
    {'file': '01_t1_data.csv', 'label': 'A', 'raw_code': 'sa7mprty',
     'contrib_code': 'sa7mprty', 'payoff_code': 'sa7mprty', 'treatment': 1},
    {'file': '03_t2_data.csv', 'label': 'C', 'raw_code': 'z8dowljr',
     'contrib_code': 'irrzlgk2', 'payoff_code': 'z8dowljr', 'treatment': 2},
    {'file': '05_t1_data.csv', 'label': 'E', 'raw_code': 'umbzdj98',
     'contrib_code': 'umbzdj98', 'payoff_code': 'umbzdj98', 'treatment': 1},
    {'file': '08_t2_data.csv', 'label': 'H', 'raw_code': 'sylq2syi',
     'contrib_code': 'sylq2syi', 'payoff_code': 'sylq2syi', 'treatment': 2},
    {'file': '11_t1_data.csv', 'label': 'R', 'raw_code': '6sdkxl2q',
     'contrib_code': '6sdkxl2q', 'payoff_code': '6sdkxl2q', 'treatment': 1},
]

PASS_COUNT = 0
FAIL_COUNT = 0


# =====
# Main
# =====
def main():
    """Run all verification checks."""
    print("=" * 70)
    print("DATA PIPELINE VERIFICATION — Raw → Derived → Summary Tables")
    print("=" * 70)

    verify_row_counts()
    verify_session_filtering()
    verify_player_contributions()
    verify_player_payoffs()
    verify_player_demographics()
    verify_aggregate_stats()
    verify_treatment_assignment()

    print("\n" + "=" * 70)
    print(f"RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    print("=" * 70)
    return FAIL_COUNT == 0


# =====
# Row count verification
# =====
def verify_row_counts():
    """Verify fundamental row counts across all data sources."""
    header("Row Count Verification")
    contrib = load_contributions()
    payoffs = load_payoffs()
    raw = load_raw_data()

    check("Total player-rounds", len(contrib), 3520)
    check("T1 player-rounds", len(contrib[contrib['treatment'] == 1]), 1760)
    check("T2 player-rounds", len(contrib[contrib['treatment'] == 2]), 1760)
    check("Total participants (payoffs)", len(payoffs), 160)
    check("Total participants (raw)", len(raw), 160)
    check("Unique sessions", contrib['session_code'].nunique(), 10)

    for sg, rounds in SUPERGAME_ROUNDS.items():
        seg = f'supergame{sg}'
        expected = 160 * rounds
        actual = len(contrib[contrib['segment'] == seg])
        check(f"  {seg} rows (160 × {rounds})", actual, expected)


# =====
# Session 03_t2 filtering
# =====
def verify_session_filtering():
    """Verify 03_t2 raw data correctly drops irrzlgk2."""
    header("Session 03_t2 Filtering")
    raw_full = pd.read_csv(
        RAW_DIR / '03_t2_data.csv', encoding='utf-8-sig'
    )
    raw_filtered = load_raw_data()
    s03 = raw_filtered[raw_filtered['session.code'] == 'z8dowljr']

    check("03_t2 raw file total rows", len(raw_full), 32)
    check("03_t2 filtered to z8dowljr only", len(s03), 16)
    check(
        "irrzlgk2 absent from load_raw_data()",
        'irrzlgk2' not in raw_filtered['session.code'].values,
        True,
    )


# =====
# Player contribution tracing
# =====
def verify_player_contributions():
    """Trace individual contributions from raw → derived for test players."""
    header("Player Contribution Tracing (Raw → Derived)")
    contrib = load_contributions()

    for p in TEST_PLAYERS:
        raw = pd.read_csv(RAW_DIR / p['file'], encoding='utf-8-sig')
        if p['file'] == '03_t2_data.csv':
            raw = raw[raw['session.code'] == 'z8dowljr']

        raw_player = raw[raw['participant.label'] == p['label']]
        if len(raw_player) == 0:
            fail(f"  {p['label']}@{p['raw_code']}: not found in raw")
            continue

        raw_row = raw_player.iloc[0]
        derived_player = contrib[
            (contrib['session_code'] == p['contrib_code'])
            & (contrib['label'] == p['label'])
        ]

        trace_contributions_for_player(p, raw_row, derived_player)


def trace_contributions_for_player(player, raw_row, derived_df):
    """Compare raw contribution columns to derived CSV rows."""
    name = f"{player['label']}@{player['raw_code']}"
    mismatches = 0

    for sg, rounds in SUPERGAME_ROUNDS.items():
        seg = f'supergame{sg}'
        for r in range(1, rounds + 1):
            raw_col = f'{seg}.{r}.player.contribution'
            raw_val = raw_row.get(raw_col)
            derived_row = derived_df[
                (derived_df['segment'] == seg) & (derived_df['round'] == r)
            ]
            if len(derived_row) == 0:
                fail(f"  {name} {seg} R{r}: missing from derived")
                mismatches += 1
                continue

            derived_val = derived_row.iloc[0]['contribution']
            if not vals_match(raw_val, derived_val):
                fail(
                    f"  {name} {seg} R{r}: "
                    f"raw={raw_val} ≠ derived={derived_val}"
                )
                mismatches += 1

    total_rounds = sum(SUPERGAME_ROUNDS.values())
    if mismatches == 0:
        check(f"  {name}: all {total_rounds} contributions match", True, True)


# =====
# Payoff tracing
# =====
def verify_player_payoffs():
    """Trace payoffs from raw → derived for test players."""
    header("Player Payoff Tracing (Raw → Derived)")
    payoffs = load_payoffs()
    contrib = load_contributions()

    for p in TEST_PLAYERS:
        raw = pd.read_csv(RAW_DIR / p['file'], encoding='utf-8-sig')
        if p['file'] == '03_t2_data.csv':
            raw = raw[raw['session.code'] == 'z8dowljr']

        raw_row = raw[raw['participant.label'] == p['label']].iloc[0]
        pay_row = payoffs[
            (payoffs['session_code'] == p['payoff_code'])
            & (payoffs['participant_label'] == p['label'])
        ]
        name = f"{p['label']}@{p['raw_code']}"

        if len(pay_row) == 0:
            fail(f"  {name}: missing from payoffs CSV")
            continue

        pay_row = pay_row.iloc[0]

        # Check per-supergame payoffs match sum of rounds in contributions
        for sg in range(1, 6):
            seg = f'supergame{sg}'
            sg_contrib = contrib[
                (contrib['session_code'] == p['contrib_code'])
                & (contrib['label'] == p['label'])
                & (contrib['segment'] == seg)
            ]
            sum_payoff = sg_contrib['payoff'].sum()
            derived_sg = pay_row[f'sg{sg}_payoff']
            check(
                f"  {name} SG{sg} payoff sum",
                round(sum_payoff, 2),
                round(derived_sg, 2),
            )

        # total_payoff is the payment-selected payoff (from oTree
        # finalresults), NOT the sum of all 5 supergames. Verify it
        # matches the raw finalresults.1.player.final_payoff.
        derived_total = pay_row['total_payoff']
        raw_final = raw_row.get('finalresults.1.player.final_payoff')
        if pd.notna(raw_final):
            check(
                f"  {name} total_payoff matches raw final_payoff",
                round(derived_total, 2),
                round(float(raw_final), 2),
            )
            expected_dollars = float(raw_final) * POINTS_TO_DOLLARS + PARTICIPATION_FEE
            derived_dollars = derived_total * POINTS_TO_DOLLARS + PARTICIPATION_FEE
            check(
                f"  {name} dollar earnings",
                round(derived_dollars, 2),
                round(expected_dollars, 2),
            )


# =====
# Demographics tracing
# =====
def verify_player_demographics():
    """Verify survey data flows correctly from raw to summary."""
    header("Demographics Tracing (Raw → load_raw_data)")
    raw_all = load_raw_data()

    for p in TEST_PLAYERS:
        raw_file = pd.read_csv(RAW_DIR / p['file'], encoding='utf-8-sig')
        if p['file'] == '03_t2_data.csv':
            raw_file = raw_file[raw_file['session.code'] == 'z8dowljr']

        raw_row = raw_file[raw_file['participant.label'] == p['label']].iloc[0]
        loaded_row = raw_all[
            (raw_all['participant.label'] == p['label'])
            & (raw_all['session.code'] == p['raw_code'])
        ]
        name = f"{p['label']}@{p['raw_code']}"

        if len(loaded_row) == 0:
            fail(f"  {name}: not found in load_raw_data()")
            continue

        loaded_row = loaded_row.iloc[0]
        for q in ['q1', 'q2', 'q3', 'q5', 'q6']:
            col = f'finalresults.1.player.{q}'
            raw_val = normalize_val(raw_row.get(col, ''))
            loaded_val = normalize_val(loaded_row.get(col, ''))
            check(f"  {name} {q}", loaded_val, raw_val)


# =====
# Aggregate statistics verification
# =====
def verify_aggregate_stats():
    """Recompute key aggregate stats and compare to .tex table values."""
    header("Aggregate Statistics Verification (Derived → Tables)")
    contrib = load_contributions()
    payoffs = load_payoffs()

    # Contribution descriptive: T1 SG1
    t1_sg1 = contrib[
        (contrib['treatment'] == 1) & (contrib['segment'] == 'supergame1')
    ]
    check_approx("  T1 SG1 mean contribution", t1_sg1['contribution'].mean(), 17.44)
    check_approx("  T1 SG1 SD contribution", t1_sg1['contribution'].std(), 8.35)
    check("  T1 SG1 N", len(t1_sg1), 240)

    # T2 SG5
    t2_sg5 = contrib[
        (contrib['treatment'] == 2) & (contrib['segment'] == 'supergame5')
    ]
    check_approx("  T2 SG5 mean contribution", t2_sg5['contribution'].mean(), 22.02)

    # Payoff: verify counts and sanity ranges
    t1_pay = payoffs[payoffs['treatment'] == 1]['total_payoff']
    t2_pay = payoffs[payoffs['treatment'] == 2]['total_payoff']
    check("  T1 participant count", len(t1_pay), 80)
    check("  T2 participant count", len(t2_pay), 80)
    # Payoffs should be positive and in a reasonable range for this game
    check("  T1 mean payoff > 0", t1_pay.mean() > 0, True)
    check("  T2 mean payoff > 0", t2_pay.mean() > 0, True)
    check("  T1 mean payoff < 500", t1_pay.mean() < 500, True)
    check("  T2 mean payoff < 500", t2_pay.mean() < 500, True)

    # Extreme contributions: % zero and % max for T1 overall
    t1 = contrib[contrib['treatment'] == 1]
    pct_zero = (t1['contribution'] == 0).mean() * 100
    pct_max = (t1['contribution'] == 25).mean() * 100
    check(f"  T1 % zero ({pct_zero:.1f}%) is reasonable", pct_zero < 30, True)
    check(f"  T1 % max ({pct_max:.1f}%) is reasonable", pct_max > 20, True)

    # Group cooperation: T1 SG1 should be ~0.70
    t1_sg1_coop = t1_sg1['contribution'].mean() / 25
    check_approx("  T1 SG1 cooperation rate", t1_sg1_coop, 0.70, tol=0.05)


# =====
# Treatment assignment consistency
# =====
def verify_treatment_assignment():
    """Verify treatment codes are consistent across all data sources.

    Note: contributions.csv uses irrzlgk2 for session 03, while
    payoffs.csv uses z8dowljr. We check each CSV with its own codes.
    """
    header("Treatment Assignment Consistency")
    contrib = load_contributions()
    payoffs = load_payoffs()

    for code, treatment in SESSION_CODE_TO_TREATMENT.items():
        c_treat = contrib[contrib['session_code'] == code]['treatment']
        if len(c_treat) > 0:
            check(
                f"  {code} contrib treatment",
                int(c_treat.unique()[0]),
                treatment,
            )

        p_treat = payoffs[payoffs['session_code'] == code]['treatment']
        if len(p_treat) > 0:
            check(
                f"  {code} payoffs treatment",
                int(p_treat.unique()[0]),
                treatment,
            )

    # Explicitly verify session 03 code mismatch is documented
    check(
        "  Session 03: contrib uses irrzlgk2",
        'irrzlgk2' in contrib['session_code'].values,
        True,
    )
    check(
        "  Session 03: payoffs uses z8dowljr",
        'z8dowljr' in payoffs['session_code'].values,
        True,
    )


# =====
# Helpers
# =====
def vals_match(a, b):
    """Compare two values, handling NaN."""
    if pd.isna(a) and pd.isna(b):
        return True
    try:
        return abs(float(a) - float(b)) < 0.01
    except (TypeError, ValueError):
        return str(a) == str(b)


def normalize_val(val):
    """Normalize a value for comparison — handles float/int string mismatch.

    pd.concat can turn '21' into 21.0; we normalize both to '21'.
    """
    s = str(val).strip()
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return s
    except (ValueError, TypeError):
        return s


def header(title):
    """Print a section header."""
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


def check(label, actual, expected):
    """Assert equality and print result."""
    global PASS_COUNT, FAIL_COUNT
    if actual == expected:
        PASS_COUNT += 1
        print(f"  PASS  {label}")
    else:
        FAIL_COUNT += 1
        print(f"  FAIL  {label}: got {actual}, expected {expected}")


def check_approx(label, actual, expected, tol=0.1):
    """Assert approximate equality."""
    global PASS_COUNT, FAIL_COUNT
    if abs(float(actual) - float(expected)) <= tol:
        PASS_COUNT += 1
        print(f"  PASS  {label} ({actual:.2f} ≈ {expected})")
    else:
        FAIL_COUNT += 1
        print(f"  FAIL  {label}: got {actual:.2f}, expected ≈{expected}")


def fail(msg):
    """Record a failure."""
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"  FAIL  {msg}")


# %%
if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
