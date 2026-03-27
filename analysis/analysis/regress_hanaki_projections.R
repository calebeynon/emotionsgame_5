# Purpose: External validation regressions — chat embedding projections -> investment
# Author: Claude Code
# Date: 2026-03-26
#
# DVs: Individual investment (Inv), Pair average investment (PairAveCho)
# IVs: 5 embedding projection scores (cooperative, promise, homogeneity, round_liar, cumulative_liar)
# FE: session_file + period
# Clustering: pair_id (session x group)

# nolint start
library(data.table)
library(fixest)

# FILE PATHS
INPUT_CSV <- "datastore/derived/hanaki_ozkes_projections.csv"
OUTPUT_DIR <- "output/tables"
OUTPUT_TEX_INV <- file.path(OUTPUT_DIR, "hanaki_external_validation_inv.tex")
OUTPUT_TEX_PAIR <- file.path(OUTPUT_DIR, "hanaki_external_validation_pair.tex")

# PROJECTION COLUMN NAMES
PROJ_VARS <- c(
    "proj_cooperative", "proj_promise", "proj_homogeneity",
    "proj_round_liar", "proj_cumulative_liar"
)

# =====
# Main function
# =====
main <- function() {
    dt <- load_and_prepare_data(INPUT_CSV)
    validate_data(dt)

    models_inv <- run_all_specs(dt, "Inv")
    models_pair <- run_all_specs(dt, "PairAveCho")

    report_sample_sizes(models_inv, models_pair)

    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
    export_table(models_inv, OUTPUT_TEX_INV, "Individual Investment")
    export_table(models_pair, OUTPUT_TEX_PAIR, "Pair Average Investment")

    cat("Exported:", OUTPUT_TEX_INV, "\n")
    cat("Exported:", OUTPUT_TEX_PAIR, "\n")
}

# =====
# Data loading and preparation
# =====
load_and_prepare_data <- function(filepath) {
    dt <- as.data.table(read.csv(filepath))
    dt[, pair_id := paste(session_file, group, sep = "_")]
    dt[, period := as.factor(period)]
    return(dt)
}

# =====
# Data validation
# =====
validate_data <- function(dt) {
    required <- c("Inv", "PairAveCho", "pair_id", PROJ_VARS)
    missing <- setdiff(required, names(dt))
    if (length(missing) > 0) {
        stop("Missing columns: ", paste(missing, collapse = ", "))
    }
    cat("Rows:", nrow(dt), " Pairs:", uniqueN(dt$pair_id), "\n")
}

# =====
# Regression specifications
# =====
run_all_specs <- function(dt, dv) {
    univariate <- lapply(PROJ_VARS, function(pv) run_univariate(dt, dv, pv))
    multivariate <- run_multivariate(dt, dv)
    c(univariate, list(multivariate))
}

run_univariate <- function(dt, dv, proj_var) {
    fml <- as.formula(paste(dv, "~", proj_var, "| session_file + period"))
    feols(fml, data = dt, cluster = ~pair_id)
}

run_multivariate <- function(dt, dv) {
    rhs <- paste(PROJ_VARS, collapse = " + ")
    fml <- as.formula(paste(dv, "~", rhs, "| session_file + period"))
    feols(fml, data = dt, cluster = ~pair_id)
}

# =====
# Sample accountability
# =====
report_sample_sizes <- function(models_inv, models_pair) {
    cat("\n=== Sample Sizes ===\n")
    cat("  Inv models: N =", models_inv[[1]]$nobs, "\n")
    cat("  Pair models: N =", models_pair[[1]]$nobs, "\n")
}

# =====
# LaTeX export
# =====
export_table <- function(models, filepath, title) {
    headers <- c(
        "Coop", "Promise", "Homog", "Rnd Liar", "Cum Liar", "All"
    )
    etable(
        models,
        file = filepath,
        tex = TRUE,
        fitstat = c("n", "r2", "ar2"),
        dict = build_var_dict(),
        headers = headers,
        title = paste("External Validation:", title)
    )
}

build_var_dict <- function() {
    c(
        proj_cooperative = "Cooperative",
        proj_promise = "Promise",
        proj_homogeneity = "Homogeneity",
        proj_round_liar = "Round Liar",
        proj_cumulative_liar = "Cumulative Liar"
    )
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
