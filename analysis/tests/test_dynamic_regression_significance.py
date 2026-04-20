"""
Coefficient parity tests for dynamic_regression.R against Table DP1 reference.

Parses output/tables/dynamic_regression_baseline.tex (4 cols) and
output/tables/dynamic_regression_extended.tex (12 cols). Verifies that the
baseline coefficients match analysis/issues/issue_68_table_dp1_reference.txt
within 0.01 tolerance and that the extended table has 12 data columns.

Author: Claude Code (test-writer)
Date: 2026-04-19
"""

import re
import subprocess
from pathlib import Path

import pytest

# FILE PATHS
REPO_ROOT = Path(__file__).parent.parent.parent
ANALYSIS_DIR = REPO_ROOT / "analysis"
R_SCRIPT = ANALYSIS_DIR / "analysis" / "dynamic_regression.R"
BASELINE_TEX = ANALYSIS_DIR / "output" / "tables" / "dynamic_regression_baseline.tex"
EXTENDED_TEX = ANALYSIS_DIR / "output" / "tables" / "dynamic_regression_extended.tex"

# Reference coefficients from analysis/issues/issue_68_table_dp1_reference.txt
# Column order in dynamic_regression_baseline.tex (0-indexed):
#   0: T1 (mean), 1: T2 (mean), 2: T1 (min/med/max), 3: T2 (min/med/max)
# Keys are texreg display labels from build_coef_names().
TOLERANCE = 0.01

T1_MEAN_REFERENCE = {
    "Contribution$_{t-1}$":       -0.069,
    "Contribution$_{t-2}$":       -0.181,
    "Positive Deviation$_{t-1}$": -0.406,
    "Negative Deviation$_{t-1}$":  0.268,
    "Round 1":                   -12.715,
}

T2_MEAN_REFERENCE = {
    "Contribution$_{t-1}$":        0.087,
    "Contribution$_{t-2}$":       -0.242,
    "Positive Deviation$_{t-1}$": -0.263,
    "Negative Deviation$_{t-1}$":  0.553,
    "Round 1":                   -5.591,
}

T1_MINMEDMAX_REFERENCE = {
    "Contribution$_{t-1}$":  -0.056,
    "Contribution$_{t-2}$":  -0.177,
    "contmoremax$_{t-1}$":    0.064,
    "contlessmax$_{t-1}$":    0.071,
    "contmoremed$_{t-1}$":   -0.179,
    "contlessmed$_{t-1}$":    0.201,
    "contmoremin$_{t-1}$":   -0.160,
    "contlessmin$_{t-1}$":   -0.016,
    "Round 1":              -12.811,
}

REFERENCE_COLUMN_INDEX = {
    "T1_mean":       0,
    "T2_mean":       1,
    "T1_minmedmax":  2,
}


# =====
# Main function (FIRST - shows high-level flow)
# =====
def main():
    """Manual invocation: parse both .tex files and print the parsed rows."""
    baseline = parse_tex_coefficients(BASELINE_TEX, num_data_cols=4)
    extended = parse_tex_coefficients(EXTENDED_TEX, num_data_cols=12)
    print("Baseline:")
    for label, row in baseline.items():
        print(f"  {label}: {row}")
    print(f"Extended: {len(extended)} rows, first = {next(iter(extended.items()))}")


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def baseline_coefs():
    """Ensure baseline .tex exists and parse into {label: [coefs per column]}."""
    ensure_tex_outputs_current()
    if not BASELINE_TEX.exists():
        raise FileNotFoundError(
            f"Missing baseline table: {BASELINE_TEX}. "
            f"Run: cd analysis && Rscript analysis/dynamic_regression.R"
        )
    return parse_tex_coefficients(BASELINE_TEX, num_data_cols=4)


@pytest.fixture(scope="module")
def extended_coefs():
    """Parse the 12-column extended .tex into {label: [coefs per column]}."""
    ensure_tex_outputs_current()
    if not EXTENDED_TEX.exists():
        raise FileNotFoundError(
            f"Missing extended table: {EXTENDED_TEX}. "
            f"Run: cd analysis && Rscript analysis/dynamic_regression.R"
        )
    return parse_tex_coefficients(EXTENDED_TEX, num_data_cols=12)


def ensure_tex_outputs_current():
    """Re-run dynamic_regression.R only if either .tex output is missing."""
    if BASELINE_TEX.exists() and EXTENDED_TEX.exists():
        return
    if not R_SCRIPT.exists():
        raise FileNotFoundError(f"Missing R script: {R_SCRIPT}")
    result = subprocess.run(
        ["Rscript", "analysis/dynamic_regression.R"],
        cwd=ANALYSIS_DIR, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"dynamic_regression.R failed (exit {result.returncode}).\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


# =====
# .tex parsing
# =====
# Matches a numeric coefficient cell: $-0.069^{***}$ or $0.268$.
COEF_CELL_RE = re.compile(r"\$(-?\d+(?:\.\d+)?)(?:\^\{[\*]+\})?\$")


def _is_structural_line(line: str) -> bool:
    """Return True for non-data tabular lines (rules, multicol, resizebox)."""
    structural = ("\\toprule", "\\midrule", "\\bottomrule",
                  "\\cmidrule", "\\multicolumn", "\\resizebox")
    return any(tok in line for tok in structural)


def _parse_tabular_body(text: str, num_data_cols: int) -> dict:
    """Extract coefficient rows from the tabular body of a .tex file."""
    in_tabular = False
    rows = {}
    for raw in text.splitlines():
        line = raw.rstrip()
        if "\\begin{tabular}" in line:
            in_tabular = True
        elif "\\end{tabular}" in line:
            break
        elif not in_tabular or "\\" not in line or _is_structural_line(line):
            continue
        else:
            cells = split_tex_row(line)
            if cells is None or len(cells) != num_data_cols + 1:
                continue
            label = cells[0].strip()
            if label and label not in rows:
                rows[label] = [parse_cell(c) for c in cells[1:]]
    return rows


def parse_tex_coefficients(tex_path: Path, num_data_cols: int) -> dict:
    """Parse dynamic_regression_*.tex -> {coef_label: [float|None, ...]}.

    Only coefficient rows (not SE rows) are returned.
    """
    rows = _parse_tabular_body(tex_path.read_text(), num_data_cols)
    if not rows:
        raise RuntimeError(f"No coefficient rows parsed from {tex_path}")
    return rows


def split_tex_row(line: str):
    r"""Split a tabular row terminated by \\ into cells. Returns None if not a row."""
    stripped = line.strip()
    if not stripped.endswith("\\\\"):
        return None
    body = stripped[: -len("\\\\")]
    return body.split("&")


def parse_cell(cell: str):
    """Parse a single cell. Returns float for a numeric coef, or None if blank."""
    trimmed = cell.strip()
    if trimmed == "" or trimmed == "$$":
        return None
    match = COEF_CELL_RE.search(trimmed)
    if match is None:
        return None
    return float(match.group(1))


# =====
# Baseline .tex: coefficient magnitude parity with Table DP1 reference
# =====
def _check_coef(label, expected, actual, col_idx, model_label):
    """Return a failure string if actual does not match expected, else None."""
    if actual is None:
        return (f"{label}: blank cell at column {col_idx} ({model_label}); "
                f"expected {expected:+.3f}")
    if abs(actual - expected) > TOLERANCE:
        return (f"{label}: got {actual:+.3f}, expected {expected:+.3f} "
                f"(diff {actual - expected:+.3f} exceeds {TOLERANCE})")
    return None


@pytest.mark.parametrize("model_label,reference", [
    ("T1_mean",      T1_MEAN_REFERENCE),
    ("T2_mean",      T2_MEAN_REFERENCE),
    ("T1_minmedmax", T1_MINMEDMAX_REFERENCE),
])
def test_baseline_coefficients_match_reference(baseline_coefs, model_label, reference):
    """Every reference coefficient matches the parsed .tex value within 0.01."""
    col_idx = REFERENCE_COLUMN_INDEX[model_label]
    failures = []
    for label, expected in reference.items():
        if label not in baseline_coefs:
            failures.append(f"{label}: missing from baseline .tex")
            continue
        msg = _check_coef(label, expected, baseline_coefs[label][col_idx], col_idx, model_label)
        if msg:
            failures.append(msg)
    assert not failures, (
        f"{model_label} reference mismatches:\n  " + "\n  ".join(failures)
    )


def test_baseline_has_four_data_columns(baseline_coefs):
    """Baseline .tex must expose 4 model columns (T1 mean, T2 mean, T1 mmm, T2 mmm)."""
    for label, row in baseline_coefs.items():
        assert len(row) == 4, (
            f"Row '{label}' has {len(row)} columns, expected 4"
        )


# =====
# Extended .tex: structural checks for 12 columns
# =====
def test_extended_has_twelve_data_columns(extended_coefs):
    """Extended .tex must expose 12 model columns (4 baselines x 3 specs)."""
    for label, row in extended_coefs.items():
        assert len(row) == 12, (
            f"Row '{label}' in extended has {len(row)} columns, expected 12"
        )


def test_extended_col0_matches_baseline_t1_mean(baseline_coefs, extended_coefs):
    """Extended col 0 (T1 mean Base) should match baseline col 0 within 0.01."""
    for label in T1_MEAN_REFERENCE:
        base_val = baseline_coefs[label][0]
        ext_val = extended_coefs[label][0]
        assert base_val is not None and ext_val is not None, (
            f"{label}: baseline={base_val}, extended={ext_val} (expected both non-null)"
        )
        assert abs(base_val - ext_val) <= TOLERANCE, (
            f"{label}: baseline T1 mean = {base_val:+.3f} but extended col 0 = "
            f"{ext_val:+.3f} (diff exceeds {TOLERANCE})"
        )


def test_extended_col6_matches_baseline_t1_minmedmax(baseline_coefs, extended_coefs):
    """Extended col 6 (T1 min/med/max Base) should match baseline col 2 within 0.01."""
    for label in T1_MINMEDMAX_REFERENCE:
        base_val = baseline_coefs[label][2]
        ext_val = extended_coefs[label][6]
        assert base_val is not None and ext_val is not None, (
            f"{label}: baseline={base_val}, extended={ext_val} (expected both non-null)"
        )
        assert abs(base_val - ext_val) <= TOLERANCE, (
            f"{label}: baseline T1 min/med/max = {base_val:+.3f} but extended col 6 = "
            f"{ext_val:+.3f} (diff exceeds {TOLERANCE})"
        )


if __name__ == "__main__":
    main()
