# Purpose: Panel regression of embedding projections on group and others' contribution
# Author: Claude Code
# Date: 2026-03-22
#
# Mirrors the individual-contribution embedding regression (Issue #42) but uses
# group-level dependent variables:
#   1. group_contribution = own + others' contributions (0-100)
#   2. others_contribution = others' total contribution (0-75)
#
# Model: [dv] ~ proj_[x]_pr_dir_small | round + segment
# Clustering: session_code x segment x group
# N ≈ 2,298 player-rounds (Contribute rows with non-null projections)

# nolint start
library(data.table)
library(fixest)

# FILE PATHS
INPUT_CSV <- "datastore/derived/merged_panel.csv"
OUTPUT_DIR <- "output/tables"
OUTPUT_GROUP_TEX <- file.path(OUTPUT_DIR, "group_contribution_embedding_regression.tex")
OUTPUT_OTHERS_TEX <- file.path(OUTPUT_DIR, "others_contribution_embedding_regression.tex")

# PROJECTION COLUMNS (display name -> column name)
PROJECTION_COLS <- c(
    "Cooperative" = "proj_pr_dir_small",
    "Promise" = "proj_promise_pr_dir_small",
    "Homogeneity" = "proj_homog_pr_dir_small",
    "Round-liar" = "proj_rliar_pr_dir_small",
    "Cumulative-liar" = "proj_cliar_pr_dir_small"
)

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
    dt <- load_and_prepare_data(INPUT_CSV)
    validate_data(dt)

    models_group <- run_regressions(dt, "group_contribution")
    models_others <- run_regressions(dt, "others_contribution")
    report_sample_sizes(models_group, models_others)

    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
    export_tables(models_group, models_others)
    print_console_summary(models_group, "Group Contribution")
    print_console_summary(models_others, "Others' Contribution")
}

export_tables <- function(models_group, models_others) {
    export_latex_table(
        models_group, OUTPUT_GROUP_TEX,
        "Group Contribution ~ Embedding Projections"
    )
    export_latex_table(
        models_others, OUTPUT_OTHERS_TEX,
        "Others' Contribution ~ Embedding Projections"
    )
}

# =====
# Data loading and preparation
# =====
load_and_prepare_data <- function(filepath) {
    dt <- as.data.table(read.csv(filepath))
    dt <- dt[page_type == "Contribute"]
    dt <- dt[complete.cases(dt[, ..PROJECTION_COLS])]

    dt[, group_contribution := contribution + others_total_contribution]
    dt[, others_contribution := others_total_contribution]
    dt[, cluster_id := paste(session_code, segment, group, sep = "_")]

    cat(sprintf("Loaded %d player-rounds after filtering\n", nrow(dt)))
    return(dt)
}

# =====
# Data validation
# =====
validate_data <- function(dt) {
    required <- c(
        "group_contribution", "others_contribution",
        "round", "segment", "cluster_id", "word_count", PROJECTION_COLS
    )
    missing <- setdiff(required, names(dt))
    if (length(missing) > 0) {
        stop("Missing required columns: ", paste(missing, collapse = ", "))
    }

    cat("\n=== Missing Values Report ===\n")
    for (col in required) {
        n_na <- sum(is.na(dt[[col]]))
        if (n_na > 0) cat(sprintf("  %s: %d NA values\n", col, n_na))
    }
    cat("\n")
}

# =====
# Regression estimation
# =====
run_regressions <- function(dt, dv) {
    models <- lapply(PROJECTION_COLS, function(proj_col) {
        run_single_regression(dt, dv, proj_col)
    })
    names(models) <- names(PROJECTION_COLS)
    return(models)
}

run_single_regression <- function(dt, dv, proj_col) {
    formula_str <- sprintf("%s ~ %s + word_count | round + segment", dv, proj_col)
    feols(as.formula(formula_str), data = dt, cluster = ~cluster_id)
}

# =====
# Sample accountability
# =====
report_sample_sizes <- function(models_group, models_others) {
    cat("=== Sample Sizes ===\n")
    for (name in names(models_group)) {
        cat(sprintf("  %s: N = %d\n", name, models_group[[name]]$nobs))
    }
    cat("\n")
}

# =====
# LaTeX output
# =====
export_latex_table <- function(models, filepath, title) {
    model_list <- unname(models)
    label_dict <- c(
        setNames(names(PROJECTION_COLS), PROJECTION_COLS),
        word_count = "Word Count"
    )

    do.call(etable, c(
        model_list,
        list(
            file = filepath,
            tex = TRUE,
            fitstat = c("n", "r2"),
            dict = label_dict,
            headers = names(PROJECTION_COLS),
            title = title
        )
    ))
    cat("Exported:", filepath, "\n")
}

# =====
# Console summary
# =====
print_console_summary <- function(models, dv_label) {
    cat(sprintf("\n=== %s Regressions ===\n", dv_label))
    for (name in names(models)) {
        cat(sprintf("\n--- %s ---\n", name))
        print(summary(models[[name]]))
    }
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
