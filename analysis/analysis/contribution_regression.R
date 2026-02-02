# Purpose: Regression analysis of contribution behavior with promise and sucker effects
# Author: Claude Code
# Date: 2026-01-23
#
# Model: contribution ~ made_promise + is_sucker + treatment | round + segment
# Clustering: session_code + segment + group (concatenated as cluster_id)
#
# COEFFICIENT INTERPRETATION:
#   treatment: Treatment is coded as 1 or 2. The coefficient represents the effect
#              of treatment 2 relative to treatment 1 (the reference category).
#
# SUCKER DEFINITION:
#   A player is classified as a "sucker" if they contributed the maximum (25 points)
#   in a round where a groupmate broke their promise.
#
#   - is_sucker_20: Groupmate broke promise by contributing < 20 after promising.
#                   This threshold catches more promise-breakers, meaning
#                   MORE players are classified as suckers.
#
#   - is_sucker_5: Groupmate broke promise by contributing < 5 after promising.
#                  This threshold catches only extreme promise-breakers, meaning
#                  FEWER players are classified as suckers.

# nolint start
library(data.table)
library(fixest)

# FILE PATHS
INPUT_CSV <- "datastore/derived/behavior_classifications.csv"
OUTPUT_DIR <- "output/tables"
OUTPUT_TEX <- file.path(OUTPUT_DIR, "contribution_regression.tex")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
    dt <- load_and_prepare_data(INPUT_CSV)
    validate_data(dt)

    model_20 <- run_regression(dt, "is_sucker_20")
    model_5 <- run_regression(dt, "is_sucker_5")

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
    bool_cols <- c("made_promise", "is_sucker_20", "is_sucker_5")
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
        "contribution", "made_promise", "is_sucker_20",
        "is_sucker_5", "treatment", "round", "segment", "cluster_id"
    )
    missing <- setdiff(required_cols, names(dt))
    if (length(missing) > 0) {
        stop("Missing required columns: ", paste(missing, collapse = ", "))
    }

    # Check for missing values in key variables
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
run_regression <- function(dt, sucker_var) {
    # Build formula dynamically based on sucker variable
    formula_str <- sprintf(
        "contribution ~ made_promise + %s + treatment | round + segment",
        sucker_var
    )

    model <- feols(
        as.formula(formula_str),
        data = dt,
        cluster = ~cluster_id
    )

    return(model)
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
            made_promise = "Made Promise",
            is_sucker_20 = "Is Sucker (<20)",
            is_sucker_5 = "Is Sucker (<5)",
            treatment = "Treatment",
            cluster_id = "session-segment-group"
        ),
        headers = c("Sucker (<20)", "Sucker (<5)"),
        title = "Contribution Regression: Promise and Sucker Effects"
    )
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
