# Purpose: External validation regressions — CCR chat embedding projections -> effort
# Author: Claude Code
# Date: 2026-03-26
#
# DVs: effort (subject-period panel), mean_effort / min_effort (group cross-section)
# IVs: 5 embedding projection scores + log(word count) control
# FE: session + period (panel), run (period-1)
# Clustering: session level

# nolint start
library(data.table)
library(fixest)

# FILE PATHS
INPUT_PANEL <- "datastore/derived/external/ccr_projections_panel.csv"
INPUT_GROUP <- "datastore/derived/external/ccr_projections_group.csv"
OUTPUT_DIR <- "output/tables"
OUTPUT_TEX <- file.path(OUTPUT_DIR, "external_validation_ccr.tex")
OUTPUT_TEX_DETAIL <- file.path(OUTPUT_DIR, "external_validation_ccr_detailed.tex")

# PROJECTION COLUMN NAMES
PROJ_VARS <- c(
    "proj_cooperative", "proj_promise", "proj_homogeneity",
    "proj_round_liar", "proj_cumulative_liar"
)

# =====
# Main function
# =====
main <- function() {
    panel <- load_panel(INPUT_PANEL)
    group <- load_group(INPUT_GROUP)
    validate_data(panel, group)

    models_panel <- run_panel_regressions(panel)
    models_period1 <- run_period1_regressions(panel)
    models_interact <- run_interaction_regressions(panel)
    models_group_mean <- run_group_regressions(group, "mean_effort")
    models_group_min <- run_group_regressions(group, "min_effort")

    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
    export_all_tables(
        models_panel, models_period1, models_interact,
        models_group_mean, models_group_min
    )
}

# =====
# Data loading
# =====
load_panel <- function(filepath) {
    dt <- as.data.table(read.csv(filepath))
    n_raw <- nrow(dt)
    dt <- dt[!is.na(proj_cooperative)]
    cat("Panel:", nrow(dt), "rows,", uniqueN(dt$session), "sessions")
    cat(" (dropped", n_raw - nrow(dt), "rows with missing projections)\n")
    dt[, session := as.factor(session)]
    dt[, period_f := as.factor(period)]
    dt[, run := as.factor(run)]
    dt[is.na(count_words), count_words := 0]
    dt[, log_word_count := log(1 + count_words)]
    return(dt)
}

load_group <- function(filepath) {
    dt <- as.data.table(read.csv(filepath))
    n_raw <- nrow(dt)
    dt <- dt[!is.na(proj_cooperative)]
    cat("Group:", nrow(dt), "rows")
    cat(" (dropped", n_raw - nrow(dt), "with missing projections)\n")
    dt[, session := as.factor(session)]
    dt[, run := as.factor(run)]
    dt[is.na(mean_count_words), mean_count_words := 0]
    dt[, log_word_count := log(1 + mean_count_words)]
    return(dt)
}

# =====
# Data validation
# =====
validate_data <- function(panel, group) {
    required_p <- c("effort", "session", "period", "count_words", PROJ_VARS)
    required_g <- c("mean_effort", "min_effort", "mean_count_words", PROJ_VARS)
    missing_p <- setdiff(required_p, names(panel))
    missing_g <- setdiff(required_g, names(group))
    if (length(missing_p) > 0) stop("Panel missing: ", paste(missing_p, collapse = ", "))
    if (length(missing_g) > 0) stop("Group missing: ", paste(missing_g, collapse = ", "))
}

# =====
# Panel regressions: Spec A (session FE) and Spec B (session + period FE)
# =====
run_panel_regressions <- function(dt) {
    models_a <- lapply(PROJ_VARS, function(pv) run_spec_a(dt, pv))
    models_b <- lapply(PROJ_VARS, function(pv) run_spec_b(dt, pv))
    c(models_a, models_b)
}

run_spec_a <- function(dt, proj_var) {
    fml <- as.formula(paste(
        "effort ~", proj_var, "+ log_word_count | session"
    ))
    feols(fml, data = dt, cluster = ~session)
}

run_spec_b <- function(dt, proj_var) {
    fml <- as.formula(paste(
        "effort ~", proj_var, "+ log_word_count | session + period_f"
    ))
    feols(fml, data = dt, cluster = ~session)
}

# =====
# Period 1 regressions (with run FE)
# =====
run_period1_regressions <- function(dt) {
    dt1 <- dt[period == 1]
    lapply(PROJ_VARS, function(pv) run_period1_spec(dt1, pv))
}

run_period1_spec <- function(dt1, proj_var) {
    fml <- as.formula(paste(
        "effort ~", proj_var, "+ ingroup + commonknow + log_word_count | run"
    ))
    feols(fml, data = dt1, cluster = ~session)
}

# =====
# Interaction regressions
# =====
run_interaction_regressions <- function(dt) {
    ingroup_int <- run_interaction(dt, "ingroup")
    ck_int <- run_interaction(dt, "commonknow")
    c(ingroup_int, ck_int)
}

run_interaction <- function(dt, interact_var) {
    lapply(PROJ_VARS, function(pv) {
        fml <- as.formula(paste(
            "effort ~", pv, "*", interact_var,
            "+ log_word_count | session + period_f"
        ))
        feols(fml, data = dt, cluster = ~session)
    })
}

# =====
# Group-level cross-section (mean_effort or min_effort as DV)
# =====
run_group_regressions <- function(dt, dv) {
    lapply(PROJ_VARS, function(pv) run_group_spec(dt, pv, dv))
}

run_group_spec <- function(dt, proj_var, dv) {
    fml <- as.formula(paste(
        dv, "~", proj_var, "+ ingroup + commonknow + log_word_count"
    ))
    feols(fml, data = dt, cluster = ~session)
}

# =====
# LaTeX export
# =====
export_all_tables <- function(
    models_panel, models_period1, models_interact,
    models_group_mean, models_group_min
) {
    export_comparison_table(models_panel, models_group_mean, models_period1)
    export_detailed_table(models_panel, models_interact)
    export_period1_table(models_period1)
    export_interaction_table(models_interact)
    export_group_table(models_group_mean, "mean_effort")
    export_group_table(models_group_min, "min_effort")
}

export_comparison_table <- function(models_panel, models_group, models_p1) {
    # Spec B coop is models_panel[[6]], group coop is models_group[[1]]
    selected <- list(
        models_panel[[6]], models_group[[1]], models_p1[[1]]
    )
    headers <- c("Panel Coop", "Group Coop", "P1 Coop")
    etable(
        selected, file = OUTPUT_TEX, tex = TRUE,
        fitstat = c("n", "r2", "ar2"),
        dict = build_var_dict(), headers = headers,
        title = "CCR External Validation: Effort Regressions"
    )
    cat("Exported:", OUTPUT_TEX, "\n")
}

export_detailed_table <- function(models_panel, models_interact) {
    interact_coop <- list(models_interact[[1]], models_interact[[6]])
    all_models <- c(models_panel, interact_coop)
    headers <- c(
        rep("Spec A", 5), rep("Spec B", 5),
        "Coop x Ingrp", "Coop x CK"
    )
    etable(
        all_models, file = OUTPUT_TEX_DETAIL, tex = TRUE,
        fitstat = c("n", "r2", "ar2"),
        dict = build_var_dict(), headers = headers,
        title = "CCR External Validation: Detailed Panel"
    )
    cat("Exported:", OUTPUT_TEX_DETAIL, "\n")
}

export_period1_table <- function(models) {
    outfile <- sub("\\.tex$", "_period1.tex", OUTPUT_TEX)
    etable(
        models, file = outfile, tex = TRUE,
        fitstat = c("n", "r2", "ar2"),
        dict = build_var_dict(),
        title = "CCR External Validation: Period 1 Only"
    )
    cat("Exported:", outfile, "\n")
}

export_interaction_table <- function(models) {
    outfile <- sub("\\.tex$", "_interactions.tex", OUTPUT_TEX)
    etable(
        models, file = outfile, tex = TRUE,
        fitstat = c("n", "r2", "ar2"),
        dict = build_var_dict(),
        title = "CCR External Validation: Interaction Effects"
    )
    cat("Exported:", outfile, "\n")
}

export_group_table <- function(models, dv) {
    suffix <- if (dv == "min_effort") "_group_min.tex" else "_group.tex"
    outfile <- sub("\\.tex$", suffix, OUTPUT_TEX)
    title_dv <- if (dv == "min_effort") "Min Effort" else "Mean Effort"
    etable(
        models, file = outfile, tex = TRUE,
        fitstat = c("n", "r2", "ar2"),
        dict = build_var_dict(),
        title = paste("CCR External Validation: Group Cross-Section —", title_dv)
    )
    cat("Exported:", outfile, "\n")
}

build_var_dict <- function() {
    c(
        proj_cooperative = "Cooperative",
        proj_promise = "Promise",
        proj_homogeneity = "Homogeneity",
        proj_round_liar = "Round Liar",
        proj_cumulative_liar = "Cumulative Liar",
        ingroup = "In-group",
        commonknow = "Common Knowledge",
        log_word_count = "Log(Word Count)"
    )
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
