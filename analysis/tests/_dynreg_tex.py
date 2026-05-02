"""
Shared LaTeX-parsing infrastructure for dynamic regression tests.

Parses output/tables/dynamic_regression_baseline.tex and dynamic_regression_extended.tex
into coefficient / SE / stars / GoF dictionaries used by the parity and
significance test files. Also exposes the tolerance constants and the
ensure-current helper that regenerates stale .tex files.

Author: Claude Code
Date: 2026-04-20
"""

import re
import subprocess
from pathlib import Path

# FILE PATHS
REPO_ROOT = Path(__file__).parent.parent.parent
ANALYSIS_DIR = REPO_ROOT / "analysis"
R_SCRIPT = ANALYSIS_DIR / "analysis" / "dynamic_regression.R"
INPUT_CSV = ANALYSIS_DIR / "datastore" / "derived" / "dynamic_regression_panel.csv"
BASELINE_TEX = ANALYSIS_DIR / "output" / "tables" / "dynamic_regression_baseline.tex"
EXTENDED_TEX = ANALYSIS_DIR / "output" / "tables" / "dynamic_regression_extended.tex"

# 0.005 matches Stata's 3-decimal log precision; any drift >0.005 indicates a
# genuine spec mismatch. Per-coefficient overrides handle legitimate numerical
# drift on large-magnitude intercepts.
TOLERANCE = 0.005
TOLERANCE_OVERRIDES = {
    # Round 1 intercepts (~|12|) drift ~0.006 from Stata due to GMM two-step
    # optimizer differences on large-scale constants. Relative drift <0.05%.
    "Round 1": 0.01,
}
SE_TOLERANCE = 0.005
# GoF p-values are 3-digit rounded; allow 0.002 rounding slack.
GOF_TOLERANCE = 0.002

# Column order in dynamic_regression_baseline.tex (0-indexed):
#   0: IF (mean-deviation), 1: AF (mean-deviation)
# Issue #74 dropped the min/med/max columns (cols 2-3 in the prior layout).
REFERENCE_COLUMN_INDEX = {
    "IF": 0,
    "AF": 1,
}

GOF_LABELS = {
    "Observations", "AR(1) p-value", "AR(2) p-value", "Sargan p-value",
    "Peer mean pair sum = 0 (p)",
}

# Matches a numeric coefficient cell with optional stars: $-0.069^{***}$ or $0.268$.
COEF_CELL_RE = re.compile(r"\$(-?\d+(?:\.\d+)?)(?:\^\{(\*+)\})?\$")
# Matches a standard error parenthetical cell: $(0.049)$.
SE_CELL_RE = re.compile(r"\$\((-?\d+(?:\.\d+)?)\)\$")


def ensure_tex_outputs_current():
    """Re-run dynamic_regression.R if either .tex is missing or stale vs sources."""
    if not R_SCRIPT.exists():
        raise FileNotFoundError(f"Missing R script: {R_SCRIPT}")
    if not _tex_outputs_stale():
        return
    result = _run_r_script_with_timeout()
    if result.returncode != 0:
        raise RuntimeError(
            f"dynamic_regression.R failed (exit {result.returncode}).\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def _run_r_script_with_timeout():
    """Invoke Rscript with 300s timeout; convert TimeoutExpired to actionable msg."""
    try:
        return subprocess.run(
            ["Rscript", "analysis/dynamic_regression.R"],
            cwd=ANALYSIS_DIR, capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"dynamic_regression.R timed out after 300s. "
            f"Investigate hung process or data issue. cmd={exc.cmd}"
        ) from exc


def _tex_outputs_stale() -> bool:
    """Return True if any .tex output is missing or older than R_SCRIPT/INPUT_CSV."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Missing panel CSV: {INPUT_CSV}. "
            f"Run build_dynamic_regression_panel.py first."
        )
    if not BASELINE_TEX.exists() or not EXTENDED_TEX.exists():
        return True
    newest_source = max(R_SCRIPT.stat().st_mtime, INPUT_CSV.stat().st_mtime)
    oldest_tex = min(BASELINE_TEX.stat().st_mtime, EXTENDED_TEX.stat().st_mtime)
    return newest_source > oldest_tex


def _is_structural_line(line: str) -> bool:
    """Return True for non-data tabular lines (rules, multicol, resizebox)."""
    structural = ("\\toprule", "\\midrule", "\\bottomrule",
                  "\\cmidrule", "\\multicolumn", "\\resizebox")
    return any(tok in line for tok in structural)


def _iter_tabular_rows(text: str):
    """Yield raw cell-lists for each tabular data row, in order."""
    in_tabular = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if "\\begin{tabular}" in line or "\\begin{tabular*}" in line:
            in_tabular = True
            continue
        if "\\end{tabular}" in line or "\\end{tabular*}" in line:
            break
        if not in_tabular or "\\" not in line or _is_structural_line(line):
            continue
        cells = split_tex_row(line)
        if cells is None:
            continue
        # Skip header-label rows like "Model: & (1) & (2) ..." whose cells are
        # column IDs, not coefficient cells. Data labels never end in a colon.
        if cells[0].strip().endswith(":"):
            continue
        yield cells


def split_tex_row(line: str):
    r"""Split a tabular row terminated by \\ into cells. None if not a row."""
    stripped = line.strip()
    if not stripped.endswith("\\\\"):
        return None
    body = stripped[: -len("\\\\")]
    return body.split("&")


def parse_cell(cell: str):
    """Parse a single coefficient cell. Returns float or None if blank."""
    trimmed = cell.strip()
    if trimmed == "" or trimmed == "$$":
        return None
    match = COEF_CELL_RE.search(trimmed)
    if match is None:
        raise ValueError(f"Unparseable coef cell: {trimmed!r}")
    return float(match.group(1))


def parse_coef_and_stars(cell: str):
    """Parse a coefficient cell into (value, stars). Returns (None, '') if blank."""
    trimmed = cell.strip()
    if trimmed == "" or trimmed == "$$":
        return (None, "")
    match = COEF_CELL_RE.search(trimmed)
    if match is None:
        raise ValueError(f"Unparseable coef cell: {trimmed!r}")
    return (float(match.group(1)), match.group(2) or "")


def parse_se_cell(cell: str):
    """Parse a standard error cell like $(0.049)$. Returns float or None."""
    trimmed = cell.strip()
    if trimmed == "" or trimmed == "$$":
        return None
    match = SE_CELL_RE.search(trimmed)
    if match is None:
        raise ValueError(f"Unparseable SE cell: {trimmed!r}")
    return float(match.group(1))


def parse_tex_coefficients(tex_path: Path, num_data_cols: int) -> dict:
    """Parse dynamic_regression_*.tex -> {coef_label: [float|None, ...]}."""
    text = tex_path.read_text()
    rows = {}
    for cells in _iter_tabular_rows(text):
        if len(cells) != num_data_cols + 1:
            continue
        label = cells[0].strip()
        if label and label not in rows:
            rows[label] = [parse_cell(c) for c in cells[1:]]
    if not rows:
        raise RuntimeError(f"No coefficient rows parsed from {tex_path}")
    return rows


def parse_tex_with_details(tex_path: Path, num_data_cols: int) -> dict:
    """Parse .tex into {label: [(coef, se, stars), ...]} by pairing coef+SE rows."""
    text = tex_path.read_text()
    details = {}
    state = {"label": None, "coefs": None}
    for cells in _iter_tabular_rows(text):
        if len(cells) != num_data_cols + 1:
            continue
        _consume_detail_row(cells, state, details)
    if state["label"] is not None:
        details[state["label"]] = [(c, None, s) for c, s in state["coefs"]]
    if not details:
        raise RuntimeError(f"No detail rows parsed from {tex_path}")
    return details


def _consume_detail_row(cells, state, details):
    """Update state/details for one coef row (label) or SE row (blank label)."""
    label = cells[0].strip()
    if not label and state["label"] is None:
        return
    if label:
        if state["label"] is not None:
            details[state["label"]] = [(c, None, s) for c, s in state["coefs"]]
        state["label"] = label
        state["coefs"] = [parse_coef_and_stars(c) for c in cells[1:]]
    else:
        ses = [parse_se_cell(c) for c in cells[1:]]
        details[state["label"]] = [
            (coef, se, stars)
            for (coef, stars), se in zip(state["coefs"], ses)
        ]
        state["label"] = None
        state["coefs"] = None


def parse_tex_gof_rows(tex_path: Path, num_data_cols: int) -> dict:
    """Parse GoF rows into {label: [value_per_col]}. Observations are ints."""
    text = tex_path.read_text()
    gof = {}
    for cells in _iter_tabular_rows(text):
        if len(cells) != num_data_cols + 1:
            continue
        label = cells[0].strip()
        if label not in GOF_LABELS:
            continue
        gof[label] = [_parse_gof_value(c, label) for c in cells[1:]]
    if not gof:
        raise RuntimeError(f"No GoF rows parsed from {tex_path}")
    return gof


def _parse_gof_value(cell: str, label: str):
    """Parse a GoF cell. Observations are ints (strip commas), p-values floats."""
    trimmed = cell.strip()
    if trimmed == "" or trimmed == "$$":
        return None
    match = re.search(r"\$(-?[\d,]+(?:\.\d+)?)\$", trimmed)
    if match is None:
        return None
    raw = match.group(1).replace(",", "")
    return int(raw) if label == "Observations" else float(raw)


def tolerance_for(label: str) -> float:
    """Return the per-coefficient tolerance, using overrides where set."""
    return TOLERANCE_OVERRIDES.get(label, TOLERANCE)


def check_coef(label, expected, actual, col_idx, model_label):
    """Return a failure string if actual does not match expected, else None."""
    tol = tolerance_for(label)
    if actual is None:
        return (f"{label}: blank cell at column {col_idx} ({model_label}); "
                f"expected {expected:+.3f}")
    if abs(actual - expected) > tol:
        return (f"{label}: got {actual:+.3f}, expected {expected:+.3f} "
                f"(diff {actual - expected:+.3f} exceeds {tol})")
    return None
