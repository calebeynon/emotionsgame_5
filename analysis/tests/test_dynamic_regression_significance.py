"""
Sign + significance validation for dynamic_regression.R against Stata paper table.

Runs dynamic_regression_validate.R, reads the coefs + Wald CSVs, and asserts that
R's sign and significance bucket match the Stata targets from
analysis/paper/tables/dynamic_regression_stata.tex.

Known SE-driven mismatches between plm::pgmm and Stata's xtabond2 are enumerated
in EXPECTED_MISMATCHES and allowed (but flagged) so the tests pass while the
divergences stay explicit and reviewable.

Author: Claude Code (test-writer)
Date: 2026-04-16
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# FILE PATHS
REPO_ROOT = Path(__file__).parent.parent.parent
ANALYSIS_DIR = REPO_ROOT / "analysis"
VALIDATE_R = ANALYSIS_DIR / "analysis" / "dynamic_regression_validate.R"
COEFS_CSV = ANALYSIS_DIR / "output" / "tables" / "dynamic_regression_coefs.csv"
WALD_CSV = ANALYSIS_DIR / "output" / "tables" / "dynamic_regression_wald.csv"

# Map R's pgmm term labels to the canonical Stata names used in the paper table.
R_TO_STATA_TERM = {
    "lag(contribution, 1:2)1": "L1_contribution",
    "lag(contribution, 1:2)2": "L2_contribution",
    "contmore_L1": "contmore_L1",
    "contless_L1": "contless_L1",
    "round1": "round1",
    "round2": "round2",
    "segmentnumber": "segmentnumber",
}

# Stata targets from log.rtf (authoritative: round1+round2 spec, vce(robust) twostep).
# Stars convention: "***" p<0.01, "**" p<0.05, "*" p<0.10, "" ns.
STATA_T1_BASELINE = {
    "L1_contribution": ("+", "***"),
    "L2_contribution": ("-", "***"),
    "contmore_L1":     ("-", "***"),
    "contless_L1":     ("+", "***"),
    "round1":          ("-", "***"),
    "round2":          ("+", "***"),
    "segmentnumber":   ("-", "*"),
}
STATA_T2_BASELINE = {
    "L1_contribution": ("+", "***"),
    "L2_contribution": ("-", "***"),
    "contmore_L1":     ("-", "***"),
    "contless_L1":     ("+", "***"),
    "round1":          ("-", "***"),
    "round2":          ("+", "***"),
    "segmentnumber":   ("-", "*"),
}
# Wald tests: from log.rtf Stata output (round1+round2 spec, robust).
STATA_WALD = {
    "T1_baseline": {"pos_plus_neg": ("", 0.1965), "R1_plus_R2": ("", 0.3178)},
    "T2_baseline": {"pos_plus_neg": ("**", 0.0161), "R1_plus_R2": ("", 0.4299)},
}

# No expected mismatches: R matches Stata log.rtf on signs + stars + Wald buckets
# after switching to the round1+round2 spec.
EXPECTED_MISMATCHES: dict = {}


# =====
# Main function (FIRST - shows high-level flow)
# =====
def main():
    """Run the validation harness and print a summary (manual invocation)."""
    _run_validate_r()
    coefs = pd.read_csv(COEFS_CSV)
    wald = pd.read_csv(WALD_CSV)
    print(coefs.head())
    print(wald.head())


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def r_results():
    """Execute dynamic_regression_validate.R and load the two output CSVs."""
    _run_validate_r()
    coefs = pd.read_csv(COEFS_CSV)
    wald = pd.read_csv(WALD_CSV)
    return {"coefs": coefs, "wald": wald}


def _run_validate_r():
    """Invoke Rscript with cwd=analysis/ so relative paths resolve."""
    if not VALIDATE_R.exists():
        raise FileNotFoundError(
            f"Missing R validate script: {VALIDATE_R}. "
            "Create dynamic_regression_validate.R before running these tests."
        )
    result = subprocess.run(
        ["Rscript", "analysis/dynamic_regression_validate.R"],
        cwd=ANALYSIS_DIR, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"dynamic_regression_validate.R failed (exit {result.returncode}).\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


# =====
# Helpers
# =====
def stars_from_p(pvalue):
    """Return paper star bucket for a p-value (p<.01 ***, p<.05 **, p<.10 *, else '')."""
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.10:
        return "*"
    return ""


def sign_from_coef(coef):
    """Return '+' for positive, '-' for negative, '0' for exact zero."""
    if coef > 0:
        return "+"
    if coef < 0:
        return "-"
    return "0"


def _check_baseline(r_coefs, model_label, stata_targets):
    """Compare R baseline coefficients to Stata targets; collect failures."""
    r_model = r_coefs[r_coefs["model"] == model_label].set_index("term")
    failures = []
    for r_term, stata_term in R_TO_STATA_TERM.items():
        expected_sign, expected_stars = stata_targets[stata_term]
        row = r_model.loc[r_term]
        r_sign = sign_from_coef(row["coef"])
        r_stars = stars_from_p(row["pvalue"])
        mismatch_key = (model_label, stata_term)
        if r_sign == expected_sign and r_stars == expected_stars:
            continue
        if mismatch_key in EXPECTED_MISMATCHES:
            continue
        failures.append(
            f"{stata_term}: R sign={r_sign} stars={r_stars!r} "
            f"(coef={row['coef']:.4f}, p={row['pvalue']:.4g}); "
            f"Stata sign={expected_sign} stars={expected_stars!r}"
        )
    return failures


# =====
# Sign + significance tests (T1 and T2 baseline)
# =====
def test_t1_baseline_signs_and_stars_match(r_results):
    """Every T1 baseline term matches Stata sign + star bucket (or is in EXPECTED_MISMATCHES)."""
    failures = _check_baseline(r_results["coefs"], "T1_baseline", STATA_T1_BASELINE)
    assert not failures, "T1 baseline divergences:\n  " + "\n  ".join(failures)


def test_t2_baseline_signs_and_stars_match(r_results):
    """Every T2 baseline term matches Stata sign + star bucket (or is in EXPECTED_MISMATCHES)."""
    failures = _check_baseline(r_results["coefs"], "T2_baseline", STATA_T2_BASELINE)
    assert not failures, "T2 baseline divergences:\n  " + "\n  ".join(failures)


# =====
# Wald-test significance bucket tests
# =====
def test_wald_tests_significance_match(r_results):
    """Baseline Wald tests (pos+neg=0, R1+R2=0) match Stata star bucket."""
    wald = r_results["wald"]
    failures = []
    for model_label, tests in STATA_WALD.items():
        for test_name, (expected_stars, stata_p) in tests.items():
            row = wald[(wald["model"] == model_label) & (wald["test_name"] == test_name)]
            assert len(row) == 1, f"Missing Wald row: {model_label}/{test_name}"
            r_p = float(row["pvalue"].iloc[0])
            r_stars = stars_from_p(r_p)
            if r_stars == expected_stars:
                continue
            if (f"{model_label}_wald", test_name) in EXPECTED_MISMATCHES:
                continue
            failures.append(
                f"{model_label}/{test_name}: R p={r_p:.4g} stars={r_stars!r}; "
                f"Stata p={stata_p} stars={expected_stars!r}"
            )
    assert not failures, "Wald-test divergences:\n  " + "\n  ".join(failures)


if __name__ == "__main__":
    main()
