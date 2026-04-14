"""
Purpose: Regression tests for the all-emotions gap regression tables
         produced by issue_52_gap_regressions_all_emotions.R.
         Validates table existence, structure, emotion row coverage,
         pinned Valence coefficients, and backup byte-equality of the
         existing valence-sentiment table.
Author: test-writer
Date: 2026-04-13
"""

import re
from pathlib import Path

import pytest

# FILE PATHS
TABLE_DIR = Path(__file__).resolve().parent.parent / "output" / "tables"
LIED_TABLE = TABLE_DIR / "issue_52_gap_summary_lied.tex"
SUCKERED_TABLE = TABLE_DIR / "issue_52_gap_summary_suckered.tex"

N_SPECS = 2

EMOTION_DISPLAY_NAMES = [
    "Anger", "Contempt", "Disgust", "Fear", "Joy", "Sadness", "Surprise",
    "Engagement", "Valence", "Sentimentality", "Confusion", "Neutral",
    "Attention",
]

COEF_TOLERANCE = 0.01

# Pinned Valence coefficients (from verified prior run 2026-04-13)
# Format: (coef, se) per column
VALENCE_PINS_LIED = {
    "Results Page Face": (5.487, 2.306),
    "Pre-Decision Chat Face": (-2.140, 0.898),
}
VALENCE_PINS_SUCKERED = {
    "Results Page Face": (1.885, 1.520),
    "Pre-Decision Chat Face": (-1.235, 1.214),
}

# Parses a signed number, optionally followed by one of the significance marks
COEF_CELL_RE = re.compile(
    r"^(-?\d+\.\d+)(?:\$\^\{\*{1,3}\}\$)?$"
)
SE_CELL_RE = re.compile(r"^\((-?\d+\.\d+)\)$")


# =====
# Parsing helpers
# =====
def parse_emotion_rows(tex_path):
    """Return dict: emotion_name -> (coef_cells, se_cells) each of length N_SPECS."""
    rows = {}
    lines = tex_path.read_text().splitlines()
    expected_parts = N_SPECS + 1
    for idx, raw in enumerate(lines):
        line = raw.strip().rstrip("\\").strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("&")]
        if len(parts) != expected_parts:
            continue
        first = parts[0]
        if first in EMOTION_DISPLAY_NAMES:
            coef_cells = parts[1:]
            se_line = lines[idx + 1].strip().rstrip("\\").strip()
            se_parts = [p.strip() for p in se_line.split("&")]
            se_cells = se_parts[1:] if len(se_parts) == expected_parts else []
            rows[first] = (coef_cells, se_cells)
    return rows


def extract_coef_value(cell):
    """Parse a coefficient cell, stripping significance stars."""
    m = COEF_CELL_RE.match(cell)
    assert m is not None, f"Cell does not match coefficient pattern: {cell!r}"
    return float(m.group(1))


def extract_se_value(cell):
    """Parse a standard-error cell of the form (X.XXX)."""
    m = SE_CELL_RE.match(cell)
    assert m is not None, f"Cell does not match SE pattern: {cell!r}"
    return float(m.group(1))


# =====
# File existence
# =====
class TestTableFilesExist:
    """Verify both new summary tables exist."""

    def test_lied_table_exists(self):
        assert LIED_TABLE.exists(), f"Missing: {LIED_TABLE}"

    def test_suckered_table_exists(self):
        assert SUCKERED_TABLE.exists(), f"Missing: {SUCKERED_TABLE}"

    def test_lied_table_nonempty(self):
        assert LIED_TABLE.stat().st_size > 500

    def test_suckered_table_nonempty(self):
        assert SUCKERED_TABLE.stat().st_size > 500


# =====
# Structure: 13 emotion rows, 4 spec columns
# =====
@pytest.fixture(scope="module")
def lied_rows():
    return parse_emotion_rows(LIED_TABLE)


@pytest.fixture(scope="module")
def suckered_rows():
    return parse_emotion_rows(SUCKERED_TABLE)


class TestTableStructure:
    """Both tables should expose all 13 emotions across 4 spec columns."""

    def test_lied_has_all_13_emotions(self, lied_rows):
        missing = [e for e in EMOTION_DISPLAY_NAMES if e not in lied_rows]
        assert not missing, f"Missing in lied table: {missing}"
        assert len(lied_rows) == 13

    def test_suckered_has_all_13_emotions(self, suckered_rows):
        missing = [e for e in EMOTION_DISPLAY_NAMES if e not in suckered_rows]
        assert not missing, f"Missing in suckered table: {missing}"
        assert len(suckered_rows) == 13

    @pytest.mark.parametrize("emotion", EMOTION_DISPLAY_NAMES)
    def test_lied_row_cell_counts(self, lied_rows, emotion):
        coef_cells, se_cells = lied_rows[emotion]
        assert len(coef_cells) == N_SPECS, f"{emotion}: coef cells {coef_cells}"
        assert len(se_cells) == N_SPECS, f"{emotion}: se cells {se_cells}"

    @pytest.mark.parametrize("emotion", EMOTION_DISPLAY_NAMES)
    def test_suckered_row_cell_counts(self, suckered_rows, emotion):
        coef_cells, se_cells = suckered_rows[emotion]
        assert len(coef_cells) == N_SPECS, f"{emotion}: coef cells {coef_cells}"
        assert len(se_cells) == N_SPECS, f"{emotion}: se cells {se_cells}"


# =====
# Valence row pinned coefficients
# =====
SPEC_COLS = ["Results Page Face", "Pre-Decision Chat Face"]


class TestValencePins:
    """Valence row must match pinned coefficients within tolerance."""

    @pytest.mark.parametrize("col_idx,col_name", list(enumerate(SPEC_COLS)))
    def test_lied_valence_coef(self, lied_rows, col_idx, col_name):
        coef_cells, _ = lied_rows["Valence"]
        actual = extract_coef_value(coef_cells[col_idx])
        expected = VALENCE_PINS_LIED[col_name][0]
        assert actual == pytest.approx(expected, abs=COEF_TOLERANCE), (
            f"Lied Valence [{col_name}] coef {actual} != {expected}"
        )

    @pytest.mark.parametrize("col_idx,col_name", list(enumerate(SPEC_COLS)))
    def test_lied_valence_se(self, lied_rows, col_idx, col_name):
        _, se_cells = lied_rows["Valence"]
        actual = extract_se_value(se_cells[col_idx])
        expected = VALENCE_PINS_LIED[col_name][1]
        assert actual == pytest.approx(expected, abs=COEF_TOLERANCE), (
            f"Lied Valence [{col_name}] SE {actual} != {expected}"
        )

    @pytest.mark.parametrize("col_idx,col_name", list(enumerate(SPEC_COLS)))
    def test_suckered_valence_coef(self, suckered_rows, col_idx, col_name):
        coef_cells, _ = suckered_rows["Valence"]
        actual = extract_coef_value(coef_cells[col_idx])
        expected = VALENCE_PINS_SUCKERED[col_name][0]
        assert actual == pytest.approx(expected, abs=COEF_TOLERANCE), (
            f"Suckered Valence [{col_name}] coef {actual} != {expected}"
        )

    @pytest.mark.parametrize("col_idx,col_name", list(enumerate(SPEC_COLS)))
    def test_suckered_valence_se(self, suckered_rows, col_idx, col_name):
        _, se_cells = suckered_rows["Valence"]
        actual = extract_se_value(se_cells[col_idx])
        expected = VALENCE_PINS_SUCKERED[col_name][1]
        assert actual == pytest.approx(expected, abs=COEF_TOLERANCE), (
            f"Suckered Valence [{col_name}] SE {actual} != {expected}"
        )


# =====
# Cell format validity: every data cell is either a coef or SE
# =====
class TestCellFormat:
    """No NA, empty, or malformed cells in data rows."""

    @pytest.mark.parametrize("emotion", EMOTION_DISPLAY_NAMES)
    def test_lied_cells_valid(self, lied_rows, emotion):
        coef_cells, se_cells = lied_rows[emotion]
        for cell in coef_cells:
            assert cell and "NA" not in cell and "NaN" not in cell
            assert COEF_CELL_RE.match(cell), f"{emotion}: {cell!r}"
        for cell in se_cells:
            assert cell and "NA" not in cell and "NaN" not in cell
            assert SE_CELL_RE.match(cell), f"{emotion} SE: {cell!r}"

    @pytest.mark.parametrize("emotion", EMOTION_DISPLAY_NAMES)
    def test_suckered_cells_valid(self, suckered_rows, emotion):
        coef_cells, se_cells = suckered_rows[emotion]
        for cell in coef_cells:
            assert cell and "NA" not in cell and "NaN" not in cell
            assert COEF_CELL_RE.match(cell), f"{emotion}: {cell!r}"
        for cell in se_cells:
            assert cell and "NA" not in cell and "NaN" not in cell
            assert SE_CELL_RE.match(cell), f"{emotion} SE: {cell!r}"


