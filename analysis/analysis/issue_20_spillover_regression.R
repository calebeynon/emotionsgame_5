# Purpose: Cross-segment spillover regression - does being suckered in a prior
#          segment affect contributions in subsequent segments?
# Author: Claude Code
# Date: 2026-02-06
#
# Spec 1 (binary): contribution ~ suckered_prior_segment + treatment | round + segment
# Spec 2 (decay):  contribution ~ i(segments_since_suckered) + treatment | round + segment
#
# Two thresholds: < 20 and < 5
# Sample: Segments 2-5 only (segment 1 has no prior segment), filtered by did_sample
# Clustering: ~cluster_id (session-segment-group)

# nolint start
library(data.table)
library(fixest)

# FILE PATHS
INPUT_CSV <- "datastore/derived/issue_20_did_panel.csv"
OUTPUT_DIR <- "output/tables"
OUTPUT_TEX <- file.path(OUTPUT_DIR, "issue_20_spillover_regression.tex")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
    dt <- load_and_prepare_data(INPUT_CSV)
    validate_data(dt)

    binary_20 <- run_binary_regression(dt, "20")
    decay_20 <- run_decay_regression(dt, "20")
    binary_5 <- run_binary_regression(dt, "5")
    decay_5 <- run_decay_regression(dt, "5")

    print_summary_stats(dt)

    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
    export_latex_table(binary_20, decay_20, binary_5, decay_5, OUTPUT_TEX)
    cat("\nSpillover regression table exported to:", OUTPUT_TEX, "\n")
}

# =====
# Data loading and preparation
# =====
load_and_prepare_data <- function(filepath) {
    dt <- fread(filepath)
    convert_bool_cols(dt)
    prepare_spillover_cols(dt)
    # Filter to segments 2-5 (segment 1 has no prior segment)
    dt <- dt[segment != "supergame1"]
    return(dt)
}

convert_bool_cols <- function(dt) {
    bool_cols <- c(
        "did_sample_20", "did_sample_5",
        "suckered_prior_segment_20", "suckered_prior_segment_5"
    )
    for (col in bool_cols) {
        dt[, (col) := as.integer(as.logical(get(col)))]
    }
}

prepare_spillover_cols <- function(dt) {
    # Convert float strings (e.g., "1.0") to integer; NAs remain NA
    dt[, segments_since_suckered_20 := as.integer(segments_since_suckered_20)]
    dt[, segments_since_suckered_5 := as.integer(segments_since_suckered_5)]
}

# =====
# Data validation
# =====
validate_data <- function(dt) {
    required_cols <- c(
        "contribution", "suckered_prior_segment_20", "suckered_prior_segment_5",
        "segments_since_suckered_20", "segments_since_suckered_5",
        "did_sample_20", "did_sample_5",
        "treatment", "round", "segment", "cluster_id"
    )
    missing <- setdiff(required_cols, names(dt))
    if (length(missing) > 0) {
        stop("Missing required columns: ", paste(missing, collapse = ", "))
    }
}

# =====
# Binary spillover regression (Spec 1)
# =====
run_binary_regression <- function(dt, threshold) {
    sample_col <- paste0("did_sample_", threshold)
    prior_col <- paste0("suckered_prior_segment_", threshold)

    dt_sub <- dt[get(sample_col) == 1]

    formula_str <- sprintf(
        "contribution ~ %s + treatment | round + segment",
        prior_col
    )

    feols(as.formula(formula_str), data = dt_sub, cluster = ~cluster_id)
}

# =====
# Decay spillover regression (Spec 2)
# =====
run_decay_regression <- function(dt, threshold) {
    sample_col <- paste0("did_sample_", threshold)
    since_col <- paste0("segments_since_suckered_", threshold)

    dt_sub <- dt[get(sample_col) == 1]
    # Drop rows where segments_since is NA (never-suckered players have no decay value)
    # Use i() to create dummies for each segments-since value
    formula_str <- sprintf(
        "contribution ~ i(%s) + treatment | round + segment",
        since_col
    )

    feols(as.formula(formula_str), data = dt_sub, cluster = ~cluster_id)
}

# =====
# Summary statistics
# =====
print_summary_stats <- function(dt) {
    cat("=== Spillover Regression Summary ===\n")
    cat("Observations (segments 2-5):", nrow(dt), "\n")
    for (thresh in c("20", "5")) {
        sample_col <- paste0("did_sample_", thresh)
        prior_col <- paste0("suckered_prior_segment_", thresh)
        n_sample <- sum(dt[[sample_col]] == 1)
        n_prior <- sum(dt[[sample_col]] == 1 & dt[[prior_col]] == 1)
        cat(sprintf(
            "  Threshold %s: %d obs (%d suckered in prior segment)\n",
            thresh, n_sample, n_prior
        ))
    }
}

# =====
# LaTeX output
# =====
export_latex_table <- function(m_bin_20, m_dec_20, m_bin_5, m_dec_5, filepath) {
    etable(
        m_bin_20, m_dec_20, m_bin_5, m_dec_5,
        file = filepath,
        tex = TRUE,
        fitstat = c("n", "r2"),
        headers = c(
            "Binary (< 20)", "Decay (< 20)",
            "Binary (< 5)", "Decay (< 5)"
        ),
        title = "Cross-Segment Spillover: Effect of Prior Suckering on Contributions"
    )
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
