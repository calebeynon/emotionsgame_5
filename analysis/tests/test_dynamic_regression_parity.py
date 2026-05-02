"""
Parity tests: dynamic_regression.R vs Stata Table DP1 reference.

Verifies baseline .tex coefficients, standard errors, significance stars, and
GoF rows match analysis/issues/issue_68_table_dp1_reference.txt within tolerance.
Also cross-checks that baseline columns equal their Base-spec counterparts in
the extended .tex.

Issue #74 dropped the min/med/max columns from both tables; this file now
covers only the IF and AF mean-deviation columns.

Author: Claude Code
Date: 2026-05-01
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _dynreg_tex import (  # noqa: E402
    GOF_TOLERANCE, REFERENCE_COLUMN_INDEX, SE_TOLERANCE,
    check_coef, tolerance_for,
)

IF_REFERENCE = {
    "Contribution$_{t-1}$":       -0.069,
    "Contribution$_{t-2}$":       -0.181,
    "Above peer mean$_{t-1}$":    -0.406,
    "Below peer mean$_{t-1}$":     0.268,
    "Round 1":                   -12.715,
}

AF_REFERENCE = {
    "Contribution$_{t-1}$":        0.087,
    "Contribution$_{t-2}$":       -0.242,
    "Above peer mean$_{t-1}$":    -0.263,
    "Below peer mean$_{t-1}$":     0.553,
    "Round 1":                    -5.591,
}

# Standard errors for IF column from reference file (Stata SEs).
IF_SE_REFERENCE = {
    "Contribution$_{t-1}$":       0.027,
    "Contribution$_{t-2}$":       0.020,
    "Above peer mean$_{t-1}$":    0.049,
    "Below peer mean$_{t-1}$":    0.065,
    "Round 1":                    0.742,
}

# Significance stars for IF: *** = p<0.01, ** = p<0.05, * = p<0.1.
IF_STARS_REFERENCE = {
    "Contribution$_{t-1}$":       "***",
    "Contribution$_{t-2}$":       "***",
    "Above peer mean$_{t-1}$":    "***",
    "Below peer mean$_{t-1}$":    "***",
    "Round 1":                    "***",
}

# GoF reference values: Observations from issue_68_table_dp1_reference.txt,
# p-values from current .tex output (cross-checked against Stata within 0.01).
IF_GOF_REFERENCE = {
    "Observations":                  1520,
    "AR(1) p-value":                 0.000,
    "AR(2) p-value":                 0.183,
    "Sargan p-value":                0.231,
    "Peer mean pair sum = 0 (p)":    0.137,
}
AF_GOF_REFERENCE = {
    "Observations":                  1520,
    "AR(1) p-value":                 0.000,
    "AR(2) p-value":                 0.309,
    "Sargan p-value":                0.252,
    "Peer mean pair sum = 0 (p)":    0.006,
}


# =====
# Baseline .tex: coefficient magnitude parity with Table DP1 reference
# =====
@pytest.mark.parametrize("model_label,reference", [
    ("IF", IF_REFERENCE),
    ("AF", AF_REFERENCE),
])
def test_baseline_coefficients_match_reference(baseline_coefs, model_label, reference):
    """Every reference coefficient matches the parsed .tex value within tolerance."""
    col_idx = REFERENCE_COLUMN_INDEX[model_label]
    failures = _collect_coef_failures(baseline_coefs, reference, col_idx, model_label)
    assert not failures, (
        f"{model_label} reference mismatches:\n  " + "\n  ".join(failures)
    )


def test_baseline_has_two_data_columns(baseline_coefs):
    """Baseline .tex must expose 2 model columns (IF, AF) after issue #74."""
    for label, row in baseline_coefs.items():
        assert len(row) == 2, (
            f"Row '{label}' has {len(row)} columns, expected 2"
        )


def _collect_coef_failures(coefs, reference, col_idx, model_label):
    """Return list of per-coefficient failure strings (empty = all match)."""
    failures = []
    for label, expected in reference.items():
        if label not in coefs:
            failures.append(f"{label}: missing from .tex")
            continue
        msg = check_coef(label, expected, coefs[label][col_idx], col_idx, model_label)
        if msg:
            failures.append(msg)
    return failures


# =====
# Baseline .tex: standard errors and significance stars
# =====
def test_baseline_if_se_and_stars(baseline_details):
    """IF column SEs and stars match the Stata reference row-by-row."""
    col_idx = REFERENCE_COLUMN_INDEX["IF"]
    failures = []
    for label, expected_se in IF_SE_REFERENCE.items():
        if label not in baseline_details:
            failures.append(f"{label}: missing from baseline .tex")
            continue
        _, se, stars = baseline_details[label][col_idx]
        if se is None or abs(se - expected_se) > SE_TOLERANCE:
            failures.append(
                f"{label} SE: got {se}, expected {expected_se:.3f} "
                f"(tol {SE_TOLERANCE})"
            )
        expected_stars = IF_STARS_REFERENCE[label]
        if stars != expected_stars:
            failures.append(
                f"{label} stars: got '{stars}', expected '{expected_stars}'"
            )
    assert not failures, "IF SE/stars mismatches:\n  " + "\n  ".join(failures)


# =====
# Baseline .tex: Goodness-of-fit row assertions
# =====
def test_baseline_if_gof_rows(baseline_gof):
    """IF GoF rows (Obs, AR, Sargan, peer-mean Wald) match reference."""
    col_idx = REFERENCE_COLUMN_INDEX["IF"]
    _assert_gof_column(baseline_gof, col_idx, IF_GOF_REFERENCE, "IF")


def test_baseline_af_gof_rows(baseline_gof):
    """AF GoF rows match reference."""
    col_idx = REFERENCE_COLUMN_INDEX["AF"]
    _assert_gof_column(baseline_gof, col_idx, AF_GOF_REFERENCE, "AF")


def test_baseline_diagnostics_pass_thresholds(baseline_gof):
    """AR(2) and Sargan p-values > 0.05 across both baseline columns."""
    failures = []
    for col_idx in range(2):
        ar2 = baseline_gof["AR(2) p-value"][col_idx]
        sargan = baseline_gof["Sargan p-value"][col_idx]
        if ar2 is None or ar2 <= 0.05:
            failures.append(f"col {col_idx}: AR(2) p-value {ar2} not > 0.05")
        if sargan is None or sargan <= 0.05:
            failures.append(f"col {col_idx}: Sargan p-value {sargan} not > 0.05")
    assert not failures, "Diagnostic thresholds violated:\n  " + "\n  ".join(failures)


def _assert_gof_column(gof, col_idx, reference, model_label):
    """Validate one model column against the reference GoF dict."""
    failures = []
    for label, expected in reference.items():
        if label not in gof:
            failures.append(f"{label}: missing from .tex GoF rows")
            continue
        actual = gof[label][col_idx]
        if label == "Observations":
            if actual != expected:
                failures.append(f"{label}: got {actual}, expected {expected}")
        elif actual is None or abs(actual - expected) > GOF_TOLERANCE:
            failures.append(
                f"{label}: got {actual}, expected {expected:.3f} "
                f"(tol {GOF_TOLERANCE})"
            )
    assert not failures, f"{model_label} GoF mismatches:\n  " + "\n  ".join(failures)


# =====
# Cross-table: extended.tex Base column == baseline.tex column
# Extended layout (0-indexed): 0:IF Base, 1:IF +Chat, 2:IF +Chat+Facial,
#                              3:AF Base, 4:AF +Chat, 5:AF +Chat+Facial
# =====
def test_extended_col0_matches_baseline_if(baseline_coefs, extended_coefs):
    """Extended col 0 (IF Base) matches baseline col 0 (IF) within tolerance."""
    _assert_extended_matches_baseline(baseline_coefs, extended_coefs,
                                      IF_REFERENCE, base_col=0, ext_col=0)


def test_extended_col3_matches_baseline_af(baseline_coefs, extended_coefs):
    """Extended col 3 (AF Base) matches baseline col 1 (AF) within tolerance."""
    _assert_extended_matches_baseline(baseline_coefs, extended_coefs,
                                      AF_REFERENCE, base_col=1, ext_col=3)


def _assert_extended_matches_baseline(baseline_coefs, extended_coefs, reference,
                                      base_col, ext_col):
    """Assert baseline[base_col] == extended[ext_col] for each reference label."""
    for label in reference:
        base_val = baseline_coefs[label][base_col]
        ext_val = extended_coefs[label][ext_col]
        assert base_val is not None and ext_val is not None, (
            f"{label}: baseline={base_val}, extended={ext_val} (expected both non-null)"
        )
        tol = tolerance_for(label)
        assert abs(base_val - ext_val) <= tol, (
            f"{label}: baseline col {base_col} = {base_val:+.3f} but extended "
            f"col {ext_col} = {ext_val:+.3f} (diff exceeds {tol})"
        )
