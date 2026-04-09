"""
Purpose: Tests for Issue #54 embedding emotion regression R scripts.
    Validates script execution, output file creation, and LaTeX table structure.
Author: Claude Code
Date: 2026-04-09
"""

import re
import subprocess
from pathlib import Path

import pytest

# FILE PATHS
ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis"
TABLE_DIR = Path(__file__).resolve().parent.parent / "output" / "tables"
WORKING_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = WORKING_DIR / "datastore" / "derived" / "merged_panel.csv"

# R SCRIPT PATHS
EMOTION_SCRIPT = ANALYSIS_DIR / "issue_54_embedding_emotion_regression.R"

# EXPECTED OUTPUT FILES
EMOTION_TABLE_FILES = [
    "issue_54_emotion_valence.tex",
    "issue_54_emotion_joy.tex",
    "issue_54_emotion_anger.tex",
    "issue_54_emotion_contempt.tex",
    "issue_54_emotion_surprise.tex",
]

PROJECTION_COLS = [
    "proj_pr_dir_small", "proj_promise_pr_dir_small",
    "proj_homog_pr_dir_small", "proj_rliar_pr_dir_small",
]
EMOTION_COLS = [
    "emotion_valence", "emotion_joy", "emotion_anger",
    "emotion_contempt", "emotion_surprise",
]
MIN_TEX_LINES = 10
MIN_TEX_SIZE = 200


# =====
# Helpers
# =====
def run_r_script(script_path, timeout=180):
    """Run an R script and return the completed process."""
    return subprocess.run(
        ["Rscript", str(script_path)],
        capture_output=True, text=True,
        cwd=str(WORKING_DIR), timeout=timeout,
    )


def run_r_code(code, timeout=60):
    """Run inline R code and return the completed process."""
    return subprocess.run(
        ["Rscript", "-e", code],
        capture_output=True, text=True,
        cwd=str(WORKING_DIR), timeout=timeout,
    )


def read_tex(filename):
    """Read a .tex file from the output tables directory."""
    return (TABLE_DIR / filename).read_text()


def count_tex_columns(tex_content):
    """Count columns from the first tabular environment spec."""
    match = re.search(r"\\begin\{tabular\}\{([^}]+)\}", tex_content)
    if match:
        spec = match.group(1)
        return spec.count("c") + spec.count("l") + spec.count("r")
    return 0


# =====
# Precondition checks
# =====
class TestPreconditions:
    """Verify required input files and R scripts exist."""

    @pytest.mark.parametrize("path", [DATA_FILE, EMOTION_SCRIPT],
                             ids=["data", "emotion_script"])
    def test_required_file_exists(self, path):
        assert path.exists(), f"Missing: {path}"


# =====
# Script execution tests
# =====
class TestEmotionRegressionScript:
    """Test emotion regression R script execution and outputs."""

    @pytest.mark.slow
    def test_script_runs_successfully(self):
        result = run_r_script(EMOTION_SCRIPT)
        assert result.returncode == 0, f"stderr: {result.stderr}"

    @pytest.mark.parametrize("filename", EMOTION_TABLE_FILES)
    def test_table_valid(self, filename):
        """Each emotion .tex file exists, is non-trivial, and has tabular env."""
        path = TABLE_DIR / filename
        assert path.exists(), f"Missing output: {path}"
        assert path.stat().st_size >= MIN_TEX_SIZE
        content = path.read_text()
        assert "\\begin{tabular" in content or "\\begin{table" in content
        assert len(content.strip().split("\n")) >= MIN_TEX_LINES


# =====
# LaTeX table structure validation
# =====
class TestTableStructure:
    """Validate internal structure of generated LaTeX tables."""

    @pytest.mark.parametrize("filename", EMOTION_TABLE_FILES)
    def test_table_column_count(self, filename):
        """Each table has >= 5 columns (4 univariate + 1 combined + row label)."""
        assert count_tex_columns(read_tex(filename)) >= 5

    @pytest.mark.parametrize("filename", EMOTION_TABLE_FILES)
    def test_table_has_fit_statistics(self, filename):
        """Each table reports observations and R-squared."""
        content = read_tex(filename)
        has_n = "Observations" in content or "Obs." in content
        has_r2 = "R$^2$" in content or "R2" in content or "Within" in content
        assert has_n, f"{filename} missing sample size"
        assert has_r2, f"{filename} missing R-squared"

    @pytest.mark.parametrize("filename", EMOTION_TABLE_FILES)
    def test_emotion_table_has_controls(self, filename):
        """Emotion tables include word_count and sentiment controls."""
        content = read_tex(filename)
        assert "Word" in content or "word" in content
        assert "Sentiment" in content or "sentiment" in content or "Compound" in content


# =====
# Data pipeline validation (via R)
# =====
class TestDataPipeline:
    """Verify data loading and filtering matches expectations."""

    def test_contribute_rows_and_columns(self):
        """Input data has Contribute rows with all required columns."""
        cols_str = ", ".join(f'"{c}"' for c in PROJECTION_COLS + EMOTION_COLS)
        result = run_r_code(f"""
library(data.table)
dt <- as.data.table(read.csv("datastore/derived/merged_panel.csv"))
ct <- dt[page_type == "Contribute"]
cat("CONTRIBUTE_ROWS:", nrow(ct), "\\n")
cols <- c({cols_str})
for (col in cols) cat(sprintf("HAS_%s: %s\\n", col, col %in% names(dt)))
""")
        assert result.returncode == 0, result.stderr
        n = int(re.search(r"CONTRIBUTE_ROWS:\s+(\d+)", result.stdout).group(1))
        assert n > 2000, f"Only {n} Contribute rows"
        for col in PROJECTION_COLS + EMOTION_COLS:
            assert f"HAS_{col}: TRUE" in result.stdout

    def test_cluster_id_construction(self):
        """Cluster ID constructible from session_code, segment, group."""
        result = run_r_code("""
library(data.table)
dt <- as.data.table(read.csv("datastore/derived/merged_panel.csv"))
dt <- dt[page_type == "Contribute"]
dt[, cluster_id := paste(session_code, segment, group, sep = "_")]
cat("N_CLUSTERS:", uniqueN(dt$cluster_id), "\\n")
""")
        assert result.returncode == 0, result.stderr
        n = int(re.search(r"N_CLUSTERS:\s+(\d+)", result.stdout).group(1))
        assert n > 20, f"Only {n} clusters"



# =====
# Model specification tests (via TESTING guard)
# =====
class TestEmotionModelSpec:
    """Verify emotion regression model specifications."""

    def test_emotion_script_sources_without_error(self):
        result = run_r_code(
            'TESTING <- TRUE; source("analysis/issue_54_embedding_emotion_regression.R")'
        )
        assert result.returncode == 0, result.stderr

    def test_univariate_model_fixed_effects_and_clustering(self):
        """Univariate models use player_id + segment FE, clustered on cluster_id."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_54_embedding_emotion_regression.R")
dt <- load_and_prepare_data(INPUT_CSV)
m <- run_single_regression(dt, "emotion_valence", "proj_pr_dir_small")
fe_str <- deparse(m$fml_all$fixef)
cat("HAS_PLAYER_FE:", grepl("player_id", fe_str), "\\n")
cat("HAS_SEGMENT_FE:", grepl("segment", fe_str), "\\n")
cat("CLUSTER_VAR:", as.character(m$call$cluster), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "HAS_PLAYER_FE: TRUE" in result.stdout
        assert "HAS_SEGMENT_FE: TRUE" in result.stdout
        assert "cluster_id" in result.stdout

    def test_combined_model_includes_all_projections(self):
        """Combined model has all 4 projection coefficients."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_54_embedding_emotion_regression.R")
dt <- load_and_prepare_data(INPUT_CSV)
m <- run_combined_model(dt, "emotion_valence")
coef_names <- names(coef(m))
for (col in PROJECTION_COLS) cat(sprintf("HAS_%s: %s\\n", col, col %in% coef_names))
""")
        assert result.returncode == 0, result.stderr
        for col in PROJECTION_COLS:
            assert f"HAS_{col}: TRUE" in result.stdout


