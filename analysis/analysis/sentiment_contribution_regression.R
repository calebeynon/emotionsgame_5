# Purpose: Regression analysis of sentiment's direct effect on contributions
# Author: Claude Code
# Date: 2026-04-01
#
# Model 1 (Baseline): contribution ~ sentiment_compound_mean + treatment | round + segment
# Model 2 (Extended): contribution ~ sentiment_compound_mean + message_count + treatment | round + segment
# Clustering: session_code + segment + group (concatenated as cluster_id)
#
# COEFFICIENT INTERPRETATION:
#   sentiment_compound_mean: VADER compound sentiment score [-1, 1].
#     Positive coefficient = more positive sentiment associated with higher contributions.
#   treatment: Treatment coded as 1 or 2. Coefficient = effect of treatment 2 vs 1.
#   message_count: Number of chat messages sent by this player in this round.
#     Controls for communication volume to isolate sentiment's content effect.

# nolint start
library(data.table)
library(fixest)

# FILE PATHS
INPUT_CSV <- "datastore/derived/sentiment_scores.csv"
OUTPUT_DIR <- "output/tables"
OUTPUT_TEX <- file.path(OUTPUT_DIR, "sentiment_contribution_regression.tex")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
    dt <- load_and_prepare_data(INPUT_CSV)
    validate_data(dt)

    baseline <- run_baseline_regression(dt)
    extended <- run_extended_regression(dt)

    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
    export_latex_table(baseline, extended, OUTPUT_TEX)

    cat("Regression table exported to:", OUTPUT_TEX, "\n")
}

# =====
# Data loading and preparation
# =====
load_and_prepare_data <- function(filepath) {
    dt <- as.data.table(read.csv(filepath))
    dt[, cluster_id := paste(session_code, segment, group, sep = "_")]
    return(dt)
}

# =====
# Data validation
# =====
validate_data <- function(dt) {
    required_cols <- c(
        "contribution", "sentiment_compound_mean", "message_count",
        "treatment", "round", "segment", "cluster_id"
    )
    missing <- setdiff(required_cols, names(dt))
    if (length(missing) > 0) {
        stop("Missing required columns: ", paste(missing, collapse = ", "))
    }
    for (col in required_cols) {
        n_na <- sum(is.na(dt[[col]]))
        if (n_na > 0) {
            warning(sprintf("Column '%s' has %d missing values", col, n_na))
        }
    }
}

# =====
# Regression estimation
# =====
run_baseline_regression <- function(dt) {
    feols(
        contribution ~ sentiment_compound_mean + treatment | round + segment,
        data = dt,
        cluster = ~cluster_id
    )
}

run_extended_regression <- function(dt) {
    feols(
        contribution ~ sentiment_compound_mean + message_count + treatment | round + segment,
        data = dt,
        cluster = ~cluster_id
    )
}

# =====
# LaTeX output
# =====
export_latex_table <- function(baseline, extended, filepath) {
    etable(
        baseline, extended,
        file = filepath,
        tex = TRUE,
        fitstat = c("n", "r2"),
        dict = c(
            sentiment_compound_mean = "Sentiment (Compound)",
            message_count = "Message Count",
            treatment = "Treatment",
            cluster_id = "session-segment-group"
        ),
        headers = c("Baseline", "With Message Count"),
        title = "Sentiment-Contribution Regression"
    )
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
