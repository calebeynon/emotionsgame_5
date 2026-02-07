# Purpose: Cross-segment spillover regression - does being suckered in a prior
#          segment affect contributions in subsequent segments?
# Author: Claude Code
# Date: 2026-02-06
#
# Spec 1 (binary): contribution ~ suckered_prior_segment + treatment | round + segment
# Spec 2 (decay):  contribution ~ i(segments_since_suckered) + treatment | round + segment
#
# Two thresholds: < 20 and < 5
# Two samples per threshold: main (suckered-once only) and robust (+ always-cooperator controls)
# Sample: Segments 2-5 only (segment 1 has no prior segment), filtered by did_sample
# Clustering: ~cluster_id (session-segment-group)

# nolint start
library(data.table)
library(fixest)

# FILE PATHS
INPUT_CSV <- "datastore/derived/issue_20_did_panel.csv"
OUTPUT_DIR <- "output/tables"
OUTPUT_TEX_20 <- file.path(OUTPUT_DIR, "issue_20_spillover_regression_20.tex")
OUTPUT_TEX_5 <- file.path(OUTPUT_DIR, "issue_20_spillover_regression_5.tex")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
    dt <- load_and_prepare_data(INPUT_CSV)
    validate_data(dt)

    models_20 <- run_threshold_models(dt, "20")
    models_5 <- run_threshold_models(dt, "5")

    print_summary_stats(dt)

    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
    export_threshold_table(models_20, "20", OUTPUT_TEX_20)
    export_threshold_table(models_5, "5", OUTPUT_TEX_5)
    cat("\nSpillover tables exported to:", OUTPUT_TEX_20, "and", OUTPUT_TEX_5, "\n")
}

run_threshold_models <- function(dt, threshold) {
    main_sample <- paste0("did_sample_", threshold)
    robust_sample <- paste0("did_sample_robust_", threshold)
    list(
        bin = run_binary_regression(dt, threshold, main_sample),
        dec = run_decay_regression(dt, threshold, main_sample),
        rbin = run_binary_regression(dt, threshold, robust_sample),
        rdec = run_decay_regression(dt, threshold, robust_sample)
    )
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
        "did_sample_robust_20", "did_sample_robust_5",
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
        "did_sample_robust_20", "did_sample_robust_5",
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
run_binary_regression <- function(dt, threshold, sample_col) {
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
run_decay_regression <- function(dt, threshold, sample_col) {
    since_col <- paste0("segments_since_suckered_", threshold)
    dt_sub <- dt[get(sample_col) == 1]

    # i() creates dummies for each segments-since value
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
        print_threshold_stats(dt, thresh)
    }
}

print_threshold_stats <- function(dt, thresh) {
    prior_col <- paste0("suckered_prior_segment_", thresh)
    for (label in c("main", "robust")) {
        sample_col <- build_sample_col(thresh, label)
        n_sample <- sum(dt[[sample_col]] == 1)
        n_prior <- sum(dt[[sample_col]] == 1 & dt[[prior_col]] == 1)
        cat(sprintf(
            "  Threshold %s (%s): %d obs (%d suckered in prior segment)\n",
            thresh, label, n_sample, n_prior
        ))
    }
}

build_sample_col <- function(threshold, label) {
    if (label == "main") paste0("did_sample_", threshold)
    else paste0("did_sample_robust_", threshold)
}

# =====
# LaTeX output (one table per threshold)
# =====
export_threshold_table <- function(models, threshold, filepath) {
    etable(
        models$bin, models$dec, models$rbin, models$rdec,
        file = filepath,
        tex = TRUE,
        fitstat = c("n", "r2"),
        headers = c(
            "Binary", "Decay",
            "Binary (robust)", "Decay (robust)"
        ),
        title = sprintf(
            "Cross-Segment Spillover (threshold < %s): Effect of Prior Suckering",
            threshold
        )
    )
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
