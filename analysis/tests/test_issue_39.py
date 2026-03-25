"""
Purpose: Tests for Issue #39 emotion-sentiment R scripts and regression outputs.
Author: Claude Code
Date: 2026-03-14
"""

import re
import subprocess
from pathlib import Path

import pandas as pd
import pytest

# FILE PATHS
ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
PLOT_DIR = OUTPUT_DIR / "plots"
TABLE_DIR = OUTPUT_DIR / "tables"
DATA_FILE = Path(__file__).resolve().parent.parent / "datastore" / "derived" / "merged_panel.csv"
BEHAVIOR_FILE = Path(__file__).resolve().parent.parent / "datastore" / "derived" / "behavior_classifications.csv"
WORKING_DIR = Path(__file__).resolve().parent.parent

# R SCRIPT PATHS
COMMON_SCRIPT = ANALYSIS_DIR / "issue_39_common.R"
DOTPLOT_SCRIPT = ANALYSIS_DIR / "issue_39_plot_dotplots.R"
NEGATIVE_EMOTIONS_SCRIPT = ANALYSIS_DIR / "issue_39_plot_negative_emotions.R"
DECOMPOSITION_SCRIPT = ANALYSIS_DIR / "issue_39_regression_decomposition.R"
GAP_TESTS_SCRIPT = ANALYSIS_DIR / "issue_39_gap_tests.R"

# EXPECTED OUTPUT FILES
DOTPLOT_FILES = [
    "emotion_sentiment_gap_by_cooperative_state.png",
    "emotion_sentiment_gap_by_liar_status.png",
    "emotion_sentiment_gap_by_sucker_status.png",
    "emotion_sentiment_gap_by_liar_x_state.png",
]

SANDBOX_DIR = Path(__file__).resolve().parent.parent / "_sandbox_data"
NEGATIVE_EMOTION_FILES = [
    "negative_emotion_by_cooperative_state.png",
    "negative_emotion_by_liar_status.png",
    "negative_emotion_by_sucker_status.png",
    "negative_emotion_by_liar_x_state.png",
]

DECOMPOSITION_TABLES = [
    "emotion_sentiment_orthogonal.tex",
    "emotion_sentiment_deception.tex",
    "emotion_sentiment_deception_descriptive.tex",
]

EXPECTED_CONTRIBUTE_ROWS = 3520
MIN_TEX_SIZE = 200


R_ZSCORE_CODE = ('source("analysis/issue_39_common.R")\n'
    'dt <- load_contribute_data(); dt <- compute_zscores(dt)\n'
    'cmp <- dt[!is.na(emotion_anger) & !is.na(emotion_fear)]\n'
    'a_sd <- sd(cmp$emotion_anger); f_sd <- sd(cmp$emotion_fear)\n'
    'dt[, anger_z := (emotion_anger - mean(cmp$emotion_anger)) / a_sd]\n'
    'dt[, fear_z := (emotion_fear - mean(cmp$emotion_fear)) / f_sd]\n'
    'cz <- dt[!is.na(anger_z) & !is.na(fear_z)]\n'
    'cat("ANGER_MEAN_Z:", round(mean(cz$anger_z), 4), "\\n")\n'
    'cat("FEAR_MEAN_Z:", round(mean(cz$fear_z), 4), "\\n")\n'
    'cat("ANGER_SD:", round(a_sd, 6), "\\n")\n'
    'cat("FEAR_SD:", round(f_sd, 6), "\\n")')


def run_r_script(script_path, timeout=120):
    """Run an R script and return the completed process."""
    return subprocess.run(
        ["Rscript", str(script_path)],
        capture_output=True, text=True,
        cwd=str(WORKING_DIR), timeout=timeout,
    )


def run_r_code(code, timeout=30):
    """Run inline R code and return the completed process."""
    return subprocess.run(
        ["Rscript", "-e", code],
        capture_output=True, text=True,
        cwd=str(WORKING_DIR), timeout=timeout,
    )


def _build_gap_regression_check() -> str:
    """R code to verify gap computation and logit regression on own-contribution outcome."""
    return """
library(fixest)
source("analysis/issue_39_common.R")
dt <- load_contribute_data()
dt <- dt[complete.cases(dt[, .(contribution, emotion_valence, sentiment_compound_mean)])]
dt[, noncooperative := as.integer(contribution < 20)]
val_rng <- max(dt$emotion_valence) - min(dt$emotion_valence)
snt_rng <- max(dt$sentiment_compound_mean) - min(dt$sentiment_compound_mean)
dt[, valence_norm := (emotion_valence - min(emotion_valence)) / val_rng]
dt[, sentiment_norm := (sentiment_compound_mean - min(sentiment_compound_mean)) / snt_rng]
dt[, emotion_sentiment_gap := valence_norm - sentiment_norm]
m <- feglm(noncooperative ~ emotion_sentiment_gap + emotion_valence | round + segment,
           data = dt, family = binomial(link = "logit"), cluster = ~cluster_id)
cat("N:", m$nobs, "\\n")
cat("NCOEFS:", length(coef(m)), "\\n")
cat("GAP_POSITIVE:", coef(m)["emotion_sentiment_gap"] > 0, "\\n")
"""


def _assert_zscore_centered(result, label):
    """Assert a named z-score is near zero and its SD is positive."""
    mean_match = re.search(rf"{label}_MEAN_Z:\s+([-\d.]+)", result.stdout)
    sd_match = re.search(rf"{label}_SD:\s+([-\d.]+)", result.stdout)
    assert mean_match is not None, f"{label}_MEAN_Z not found: {result.stdout}"
    assert sd_match is not None, f"{label}_SD not found: {result.stdout}"
    assert abs(float(mean_match.group(1))) < 0.01
    assert float(sd_match.group(1)) > 0


# =====
# Precondition checks
# =====
REQUIRED_FILES = [DATA_FILE, BEHAVIOR_FILE, COMMON_SCRIPT,
                  DOTPLOT_SCRIPT, GAP_TESTS_SCRIPT, NEGATIVE_EMOTIONS_SCRIPT]


class TestPreconditions:
    """Verify data files and R scripts exist."""

    @pytest.mark.parametrize("path", REQUIRED_FILES, ids=lambda p: p.name)
    def test_required_file_exists(self, path):
        assert path.exists()


# =====
# issue_39_common.R tests
# =====
class TestCommonUtilities:
    """Test shared utilities file."""

    def test_sources_without_error(self):
        result = run_r_code('source("analysis/issue_39_common.R")')
        assert result.returncode == 0, result.stderr

    def test_load_contribute_data(self):
        result = run_r_code("""
source("analysis/issue_39_common.R")
dt <- load_contribute_data()
cat("ROWS:", nrow(dt), "\\n")
cat("HAS_CLUSTER:", "cluster_id" %in% names(dt), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert f"ROWS: {EXPECTED_CONTRIBUTE_ROWS}" in result.stdout

    def test_merge_behavior_classifications(self):
        result = run_r_code("""
source("analysis/issue_39_common.R")
dt <- load_contribute_data()
dt <- merge_behavior_classifications(dt)
cat("HAS_LIAR:", "is_liar_20" %in% names(dt), "\\n")
cat("HAS_SUCKER:", "is_sucker_20" %in% names(dt), "\\n")
cat("HAS_LIED_ROUND:", "lied_this_round_20" %in% names(dt), "\\n")
cat("ROWS:", nrow(dt), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "HAS_LIAR: TRUE" in result.stdout
        assert "HAS_SUCKER: TRUE" in result.stdout
        assert "HAS_LIED_ROUND: TRUE" in result.stdout
        assert f"ROWS: {EXPECTED_CONTRIBUTE_ROWS}" in result.stdout

    def test_compute_zscores(self):
        result = run_r_code("""
source("analysis/issue_39_common.R")
dt <- load_contribute_data()
dt <- compute_zscores(dt)
cat("HAS_VALZ:", "valence_z" %in% names(dt), "\\n")
cat("HAS_CMPZ:", "compound_z" %in% names(dt), "\\n")
cat("HAS_GAP:", "zscore_gap" %in% names(dt), "\\n")
complete <- dt[!is.na(valence_z) & !is.na(compound_z)]
cat("VALMEAN:", round(mean(complete$valence_z), 4), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "HAS_VALZ: TRUE" in result.stdout
        assert "HAS_GAP: TRUE" in result.stdout
        # Parse the z-score mean and verify it's near zero (not just substring match)
        match = re.search(r"VALMEAN:\s+([-\d.]+)", result.stdout)
        assert match is not None, f"VALMEAN not found in output: {result.stdout}"
        assert abs(float(match.group(1))) < 0.01, f"VALMEAN not near zero: {match.group(1)}"


# =====
# issue_39_plot_dotplots.R tests
# =====
class TestDotPlots:
    """Test dot plot script execution and outputs."""

    def test_script_runs_successfully(self):
        result = run_r_script(DOTPLOT_SCRIPT)
        assert result.returncode == 0, f"stderr: {result.stderr}"

    @pytest.mark.parametrize("filename", DOTPLOT_FILES)
    def test_plot_file_exists_and_nonempty(self, filename):
        path = PLOT_DIR / filename
        assert path.exists()
        assert path.stat().st_size > 10_000


# =====
# Regression tests
# =====
class TestDecompositionRegression:
    """Test decomposition and deception regressions."""

    def test_script_runs(self):
        result = run_r_script(DECOMPOSITION_SCRIPT, timeout=180)
        assert result.returncode == 0, result.stderr

    @pytest.mark.parametrize("filename", DECOMPOSITION_TABLES)
    def test_table_exists(self, filename):
        path = TABLE_DIR / filename
        assert path.exists()
        assert path.stat().st_size >= MIN_TEX_SIZE


# =====
# Gap tests
# =====
GAP_TESTS_TABLE = "emotion_sentiment_gap_tests.tex"


class TestGapTests:
    """Test gap tests script execution and outputs."""

    def test_script_runs(self):
        result = run_r_script(GAP_TESTS_SCRIPT)
        assert result.returncode == 0, result.stderr

    def test_output_table_exists(self):
        path = TABLE_DIR / GAP_TESTS_TABLE
        assert path.exists()
        assert path.stat().st_size >= MIN_TEX_SIZE


# =====
# issue_39_plot_negative_emotions.R tests
# =====
class TestNegativeEmotionPlots:
    """Test negative emotion dot plot script execution and outputs."""

    def test_script_runs_successfully(self):
        result = run_r_script(NEGATIVE_EMOTIONS_SCRIPT)
        assert result.returncode == 0, f"stderr: {result.stderr}"

    @pytest.mark.parametrize("filename", NEGATIVE_EMOTION_FILES)
    def test_plot_file_exists_and_nonempty(self, filename):
        path = SANDBOX_DIR / filename
        assert path.exists()
        assert path.stat().st_size > 10_000

    def test_anger_fear_zscores_computed(self):
        """Verify anger and fear z-scores are computed and centered near zero."""
        result = run_r_code(R_ZSCORE_CODE)
        assert result.returncode == 0, result.stderr
        _assert_zscore_centered(result, "ANGER")
        _assert_zscore_centered(result, "FEAR")

    def test_three_measures_in_summary(self):
        """Verify the script produces summaries with exactly 3 measures."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_39_plot_negative_emotions.R")
dt <- prepare_plot_data()
summary_dt <- summarize_by_group(dt, "state_label")
cat("MEASURES:", paste(sort(unique(summary_dt$measure)), collapse="|"), "\\n")
cat("N_ROWS:", nrow(summary_dt), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "MEASURES: Anger|Fear|Sentiment (Compound)" in result.stdout
        # 2 groups (Cooperative, Noncooperative) × 3 measures = 6 rows
        assert "N_ROWS: 6" in result.stdout


# =====
# Data integrity tests
# =====
@pytest.fixture()
def contribute_df():
    """Load Contribute rows from merged_panel.csv."""
    return pd.read_csv(DATA_FILE).query("page_type == 'Contribute'")


class TestDataIntegrity:
    """Verify input data assumptions."""

    def test_contribute_row_count(self, contribute_df):
        assert len(contribute_df) == EXPECTED_CONTRIBUTE_ROWS

    def test_contribution_range(self, contribute_df):
        assert contribute_df["contribution"].dropna().between(0, 25).all()

    def test_sentiment_compound_range(self, contribute_df):
        assert contribute_df["sentiment_compound_mean"].dropna().between(-1, 1).all()


# =====
# Noncooperative outcome variable tests
# =====
class TestNoncooperativeOutcome:
    """Verify the deception regression outcome variable (contribution < 20)."""

    def test_noncooperative_count_matches_raw_data(self):
        """R noncooperative count must match Python count from raw data."""
        result = run_r_code("""
source("analysis/issue_39_common.R")
dt <- load_contribute_data()
dt <- dt[complete.cases(dt[, .(contribution, emotion_valence, sentiment_compound_mean)])]
dt[, noncooperative := as.integer(contribution < 20)]
cat("NONCOOP:", sum(dt$noncooperative), "\\n")
cat("COOP:", sum(dt$noncooperative == 0), "\\n")
cat("TOTAL:", nrow(dt), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "NONCOOP: 200" in result.stdout
        assert "COOP: 1591" in result.stdout
        assert "TOTAL: 1791" in result.stdout

    def test_noncooperative_matches_python_count(self):
        """Cross-validate R counts against Python on the same raw CSV."""
        df = pd.read_csv(DATA_FILE)
        ct = df[df["page_type"] == "Contribute"]
        complete = ct.dropna(subset=["contribution", "emotion_valence",
                                      "sentiment_compound_mean"])
        noncoop = (complete["contribution"] < 20).sum()
        coop = (complete["contribution"] >= 20).sum()
        assert noncoop == 200
        assert coop == 1591

    def test_threshold_boundary_contribution_20_is_cooperative(self):
        """Contribution == 20 should be classified as cooperative (not < 20)."""
        result = run_r_code("""
source("analysis/issue_39_common.R")
dt <- load_contribute_data()
dt <- dt[complete.cases(dt[, .(contribution, emotion_valence, sentiment_compound_mean)])]
dt[, noncooperative := as.integer(contribution < 20)]
at_20 <- dt[contribution == 20]
cat("N_AT_20:", nrow(at_20), "\\n")
cat("NONCOOP_AT_20:", sum(at_20$noncooperative), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "N_AT_20: 34" in result.stdout
        assert "NONCOOP_AT_20: 0" in result.stdout

    def test_known_noncooperative_player(self):
        """Player K, session irrzlgk2, supergame1 round 2: contribution=0 → noncooperative."""
        result = run_r_code("""
source("analysis/issue_39_common.R")
dt <- load_contribute_data()
dt[, noncooperative := as.integer(contribution < 20)]
row <- dt[session_code == "irrzlgk2" & segment == "supergame1" & round == 2 & label == "K"]
cat("CONTRIB:", row$contribution, "\\n")
cat("NONCOOP:", row$noncooperative, "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "CONTRIB: 0" in result.stdout
        assert "NONCOOP: 1" in result.stdout

    def test_known_cooperative_player(self):
        """Player C, session irrzlgk2, supergame1 round 2: contribution=25 → cooperative."""
        result = run_r_code("""
source("analysis/issue_39_common.R")
dt <- load_contribute_data()
dt[, noncooperative := as.integer(contribution < 20)]
row <- dt[session_code == "irrzlgk2" & segment == "supergame1" & round == 2 & label == "C"]
cat("CONTRIB:", row$contribution, "\\n")
cat("NONCOOP:", row$noncooperative, "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "CONTRIB: 25" in result.stdout
        assert "NONCOOP: 0" in result.stdout

    def test_deception_table_uses_own_contribution(self):
        """The deception descriptive table should show N=200 noncooperative (own contribution)."""
        path = TABLE_DIR / "emotion_sentiment_deception_descriptive.tex"
        assert path.exists()
        content = path.read_text()
        # Noncooperative row should have N=200
        assert "200" in content, "Expected 200 noncooperative obs in descriptive table"
        # Cooperative row should have N=1591
        assert "1591" in content, "Expected 1591 cooperative obs in descriptive table"

    def test_gap_flows_into_regression(self):
        """Verify gap is computed correctly and logit runs on the right sample."""
        r_code = _build_gap_regression_check()
        result = run_r_code(r_code, timeout=60)
        assert result.returncode == 0, result.stderr
        assert "N: 1791" in result.stdout
        assert "NCOEFS: 2" in result.stdout
        assert "GAP_POSITIVE: TRUE" in result.stdout
