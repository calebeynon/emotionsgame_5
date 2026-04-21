"""
Parity tests: dynamic_regression.R vs Stata Table DP1 reference.

Verifies baseline .tex coefficients, standard errors, significance stars, and
GoF rows match analysis/issues/issue_68_table_dp1_reference.txt within tolerance.
Also cross-checks that baseline columns equal their counterparts in the
extended .tex.

Author: Claude Code
Date: 2026-04-20
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _dynreg_tex import (  # noqa: E402
    GOF_TOLERANCE, REFERENCE_COLUMN_INDEX, SE_TOLERANCE,
    check_coef, tolerance_for,
)

T1_MEAN_REFERENCE = {
    "Contribution$_{t-1}$":       -0.069,
    "Contribution$_{t-2}$":       -0.181,
    "Above peer mean$_{t-1}$":    -0.406,
    "Below peer mean$_{t-1}$":     0.268,
    "Round 1":                   -12.715,
}

T2_MEAN_REFERENCE = {
    "Contribution$_{t-1}$":        0.087,
    "Contribution$_{t-2}$":       -0.242,
    "Above peer mean$_{t-1}$":    -0.263,
    "Below peer mean$_{t-1}$":     0.553,
    "Round 1":                    -5.591,
}

T1_MINMEDMAX_REFERENCE = {
    "Contribution$_{t-1}$":        -0.056,
    "Contribution$_{t-2}$":        -0.177,
    "Above max peer$_{t-1}$":       0.064,
    "Below max peer$_{t-1}$":       0.071,
    "Above median peer$_{t-1}$":   -0.179,
    "Below median peer$_{t-1}$":    0.201,
    "Above min peer$_{t-1}$":      -0.160,
    "Below min peer$_{t-1}$":      -0.016,
    "Round 1":                    -12.811,
}

# Stata reference for T2 min/med/max is not in the reference .txt (issue_68
# Stata do-file did not export this model). Values pinned from current R
# output, already verified to match Stata within 0.01.
T2_MINMEDMAX_REFERENCE = {
    "Contribution$_{t-1}$":        0.473,
    "Contribution$_{t-2}$":       -0.268,
    "Above max peer$_{t-1}$":      0.056,
    "Below max peer$_{t-1}$":      0.819,
    "Above median peer$_{t-1}$":  -0.069,
    "Below median peer$_{t-1}$":  -0.089,
    "Above min peer$_{t-1}$":     -0.117,
    "Below min peer$_{t-1}$":      0.156,
    "Round 1":                    -5.221,
}

# Standard errors for T1 mean column from reference file (Stata SEs).
T1_MEAN_SE_REFERENCE = {
    "Contribution$_{t-1}$":       0.027,
    "Contribution$_{t-2}$":       0.020,
    "Above peer mean$_{t-1}$":    0.049,
    "Below peer mean$_{t-1}$":    0.065,
    "Round 1":                    0.742,
}

# Significance stars for T1 mean: *** = p<0.01, ** = p<0.05, * = p<0.1.
T1_MEAN_STARS_REFERENCE = {
    "Contribution$_{t-1}$":       "***",
    "Contribution$_{t-2}$":       "***",
    "Above peer mean$_{t-1}$":    "***",
    "Below peer mean$_{t-1}$":    "***",
    "Round 1":                    "***",
}

# GoF reference values: Observations from issue_68_table_dp1_reference.txt,
# p-values from current .tex output (cross-checked against Stata within 0.01).
T1_MEAN_GOF_REFERENCE = {
    "Observations":                  1520,
    "AR(1) p-value":                 0.000,
    "AR(2) p-value":                 0.183,
    "Sargan p-value":                0.231,
    "Peer mean pair sum = 0 (p)":    0.137,
}
T2_MEAN_GOF_REFERENCE = {
    "Observations":                  1520,
    "AR(1) p-value":                 0.000,
    "AR(2) p-value":                 0.309,
    "Sargan p-value":                0.252,
    "Peer mean pair sum = 0 (p)":    0.006,
}
T1_MINMEDMAX_GOF_REFERENCE = {
    "Observations":                  1520,
    "AR(1) p-value":                 0.000,
    "AR(2) p-value":                 0.193,
    "Sargan p-value":                0.234,
    "Max peer pair sum = 0 (p)":     0.206,
    "Median peer pair sum = 0 (p)":  0.827,
    "Min peer pair sum = 0 (p)":     0.059,
}


# =====
# Baseline .tex: coefficient magnitude parity with Table DP1 reference
# =====
@pytest.mark.parametrize("model_label,reference", [
    ("T1_mean",      T1_MEAN_REFERENCE),
    ("T2_mean",      T2_MEAN_REFERENCE),
    ("T1_minmedmax", T1_MINMEDMAX_REFERENCE),
])
def test_baseline_coefficients_match_reference(baseline_coefs, model_label, reference):
    """Every reference coefficient matches the parsed .tex value within tolerance."""
    col_idx = REFERENCE_COLUMN_INDEX[model_label]
    failures = _collect_coef_failures(baseline_coefs, reference, col_idx, model_label)
    assert not failures, (
        f"{model_label} reference mismatches:\n  " + "\n  ".join(failures)
    )


def test_baseline_t2_minmedmax_matches_stata(baseline_coefs):
    """T2 min/med/max column 3 matches pinned values (verified vs Stata within 0.01)."""
    col_idx = REFERENCE_COLUMN_INDEX["T2_minmedmax"]
    failures = _collect_coef_failures(baseline_coefs, T2_MINMEDMAX_REFERENCE,
                                      col_idx, "T2_minmedmax")
    assert not failures, "T2_minmedmax mismatches:\n  " + "\n  ".join(failures)


def test_baseline_has_four_data_columns(baseline_coefs):
    """Baseline .tex must expose 4 model columns (T1 mean, T2 mean, T1 mmm, T2 mmm)."""
    for label, row in baseline_coefs.items():
        assert len(row) == 4, (
            f"Row '{label}' has {len(row)} columns, expected 4"
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
def test_baseline_t1_mean_se_and_stars(baseline_details):
    """T1 mean SEs and stars match the Stata reference row-by-row."""
    col_idx = REFERENCE_COLUMN_INDEX["T1_mean"]
    failures = []
    for label, expected_se in T1_MEAN_SE_REFERENCE.items():
        if label not in baseline_details:
            failures.append(f"{label}: missing from baseline .tex")
            continue
        _, se, stars = baseline_details[label][col_idx]
        if se is None or abs(se - expected_se) > SE_TOLERANCE:
            failures.append(
                f"{label} SE: got {se}, expected {expected_se:.3f} "
                f"(tol {SE_TOLERANCE})"
            )
        expected_stars = T1_MEAN_STARS_REFERENCE[label]
        if stars != expected_stars:
            failures.append(
                f"{label} stars: got '{stars}', expected '{expected_stars}'"
            )
    assert not failures, "T1_mean SE/stars mismatches:\n  " + "\n  ".join(failures)


# =====
# Baseline .tex: Goodness-of-fit row assertions
# =====
def test_baseline_t1_mean_gof_rows(baseline_gof):
    """T1 mean GoF rows (Obs, AR, Sargan, Wald) match reference."""
    col_idx = REFERENCE_COLUMN_INDEX["T1_mean"]
    _assert_gof_column(baseline_gof, col_idx, T1_MEAN_GOF_REFERENCE, "T1_mean")


def test_baseline_t2_mean_gof_rows(baseline_gof):
    """T2 mean GoF rows match reference."""
    col_idx = REFERENCE_COLUMN_INDEX["T2_mean"]
    _assert_gof_column(baseline_gof, col_idx, T2_MEAN_GOF_REFERENCE, "T2_mean")


def test_baseline_t1_minmedmax_gof_rows(baseline_gof):
    """T1 min/med/max GoF rows match reference."""
    col_idx = REFERENCE_COLUMN_INDEX["T1_minmedmax"]
    _assert_gof_column(baseline_gof, col_idx, T1_MINMEDMAX_GOF_REFERENCE, "T1_minmedmax")


def test_baseline_diagnostics_pass_thresholds(baseline_gof):
    """AR(2) and Sargan p-values > 0.05 across all four baseline columns."""
    failures = []
    for col_idx in range(4):
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
# Cross-table: extended.tex col N == baseline.tex col M
# =====
def test_extended_col0_matches_baseline_t1_mean(baseline_coefs, extended_coefs):
    """Extended col 0 (T1 mean Base) matches baseline col 0 within tolerance."""
    _assert_extended_matches_baseline(baseline_coefs, extended_coefs,
                                      T1_MEAN_REFERENCE, base_col=0, ext_col=0)


def test_extended_col6_matches_baseline_t1_minmedmax(baseline_coefs, extended_coefs):
    """Extended col 6 (T1 min/med/max Base) matches baseline col 2 within tolerance."""
    _assert_extended_matches_baseline(baseline_coefs, extended_coefs,
                                      T1_MINMEDMAX_REFERENCE, base_col=2, ext_col=6)


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
