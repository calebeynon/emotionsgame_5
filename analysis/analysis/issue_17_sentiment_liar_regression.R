# Purpose: Regression analysis of sentiment-contribution relationship with liar interaction
# Author: Claude Code
# Date: 2026-01-27
#
# Models: contribution ~ sentiment_compound_mean * lied + treatment | round + segment
# Clustering: cluster_id (session-segment-group)
#
# COEFFICIENT INTERPRETATION:
#   sentiment_compound_mean: Effect of sentiment on contribution (for non-liars)
#   lied_this_period: Effect of lying on contribution (at mean sentiment)
#   interaction: How lying modifies the sentiment-contribution relationship
#   treatment: Treatment 2 effect relative to Treatment 1 (reference)
#
# LIAR DEFINITIONS:
#   _20 threshold: Made promise AND contributed < 20 (94 instances)
#   _5 threshold: Made promise AND contributed < 5 (49 instances, extreme liars only)

# nolint start
library(data.table)
library(fixest)

# FILE PATHS
INPUT_CSV <- "datastore/derived/issue_17_regression_data.csv"
OUTPUT_DIR <- "output/tables"
OUTPUT_TEX <- file.path(OUTPUT_DIR, "issue_17_sentiment_liar_regression.tex")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
    dt <- load_and_prepare_data(INPUT_CSV)
    validate_data(dt)

    model_20 <- run_regression(dt, "lied_this_period_20")
    model_5 <- run_regression(dt, "lied_this_period_5")

    report_sample_sizes(model_20, model_5)

    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
    export_latex_table(model_20, model_5, OUTPUT_TEX)

    cat("Regression table exported to:", OUTPUT_TEX, "\n")
}

# =====
# Data loading and preparation
# =====
load_and_prepare_data <- function(filepath) {
    dt <- as.data.table(read.csv(filepath))

    # Convert Python boolean strings to numeric 0/1
    bool_cols <- c("lied_this_period_20", "lied_this_period_5")
    for (col in bool_cols) {
        dt[, (col) := as.integer(get(col) == "True")]
    }

    # Create cluster_id for multi-way clustering
    dt[, cluster_id := paste(session_code, segment, group, sep = "_")]

    return(dt)
}

# =====
# Data validation
# =====
validate_data <- function(dt) {
    required_cols <- c(
        "contribution", "sentiment_compound_mean", "lied_this_period_20",
        "lied_this_period_5", "treatment", "round", "segment", "cluster_id"
    )
    missing <- setdiff(required_cols, names(dt))
    if (length(missing) > 0) {
        stop("Missing required columns: ", paste(missing, collapse = ", "))
    }

    cat("\n=== Missing Values Report ===\n")
    for (col in required_cols) {
        n_na <- sum(is.na(dt[[col]]))
        cat(sprintf("  %s: %d NA values\n", col, n_na))
    }
    cat("\n")
}

# =====
# Regression estimation
# =====
run_regression <- function(dt, liar_var) {
    formula_str <- sprintf(
        "contribution ~ sentiment_compound_mean * %s + treatment | round + segment",
        liar_var
    )
    model <- feols(as.formula(formula_str), data = dt, cluster = ~cluster_id)
    return(model)
}

# =====
# Sample accountability
# =====
report_sample_sizes <- function(model_20, model_5) {
    cat("=== Sample Sizes ===\n")
    cat(sprintf("  Liar (<20) model: N = %d\n", model_20$nobs))
    cat(sprintf("  Liar (<5) model: N = %d\n", model_5$nobs))
    cat("\n")
}

# =====
# LaTeX output
# =====
export_latex_table <- function(model_20, model_5, filepath) {
    etable(
        model_20, model_5,
        file = filepath,
        tex = TRUE,
        fitstat = c("n", "r2"),
        dict = c(
            sentiment_compound_mean = "Sentiment",
            lied_this_period_20 = "Lied (<20)",
            lied_this_period_5 = "Lied (<5)",
            treatment = "Treatment",
            "sentiment_compound_mean:lied_this_period_20" = "Sentiment x Lied",
            "sentiment_compound_mean:lied_this_period_5" = "Sentiment x Lied"
        ),
        headers = c("Liar (<20)", "Liar (<5)"),
        title = "Sentiment-Contribution Regression with Liar Interaction"
    )
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
