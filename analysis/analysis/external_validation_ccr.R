# Purpose: External validation regressions — CCR chat embedding projections -> effort
# Author: Claude Code
# Date: 2026-03-26
#
# DVs: effort (subject-period panel), mean_effort (group cross-section)
# IVs: 5 embedding projection scores + treatment dummies (ingroup, commonknow)
# FE: session, period
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
    models_site <- run_by_site_regressions(panel)
    models_group <- run_group_regressions(group)

    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
    export_all_tables(
        models_panel, models_period1, models_interact,
        models_site, models_group
    )
}

# =====
# Data loading
# =====
load_panel <- function(filepath) {
    dt <- as.data.table(read.csv(filepath))
    dt <- dt[!is.na(proj_cooperative)]
    dt[, session := as.factor(session)]
    dt[, period_f := as.factor(period)]
    dt[, run := as.factor(run)]
    cat("Panel:", nrow(dt), "rows,", uniqueN(dt$session), "sessions\n")
    return(dt)
}

load_group <- function(filepath) {
    dt <- as.data.table(read.csv(filepath))
    dt <- dt[!is.na(proj_cooperative)]
    dt[, session := as.factor(session)]
    dt[, run := as.factor(run)]
    cat("Group:", nrow(dt), "rows\n")
    return(dt)
}

# =====
# Data validation
# =====
validate_data <- function(panel, group) {
    required <- c("effort", "session", "period", PROJ_VARS)
    missing_p <- setdiff(required, names(panel))
    missing_g <- setdiff(c("mean_effort", PROJ_VARS), names(group))
    if (length(missing_p) > 0) stop("Panel missing: ", paste(missing_p, collapse = ", "))
    if (length(missing_g) > 0) stop("Group missing: ", paste(missing_g, collapse = ", "))
}

# =====
# Panel regressions (a, b for each projection)
# =====
run_panel_regressions <- function(dt) {
    models_a <- lapply(PROJ_VARS, function(pv) run_spec_a(dt, pv))
    models_b <- lapply(PROJ_VARS, function(pv) run_spec_b(dt, pv))
    multi_b <- run_multivariate_b(dt)
    c(models_a, models_b, list(multi_b))
}

run_spec_a <- function(dt, proj_var) {
    fml <- as.formula(paste("effort ~", proj_var, "| session"))
    feols(fml, data = dt, cluster = ~session)
}

run_spec_b <- function(dt, proj_var) {
    fml <- as.formula(paste(
        "effort ~", proj_var, "+ ingroup + commonknow | session + period_f"
    ))
    feols(fml, data = dt, cluster = ~session)
}

run_multivariate_b <- function(dt) {
    rhs <- paste(PROJ_VARS, collapse = " + ")
    fml <- as.formula(paste(
        "effort ~", rhs, "+ ingroup + commonknow | session + period_f"
    ))
    feols(fml, data = dt, cluster = ~session)
}

# =====
# Period 1 regressions (d)
# =====
run_period1_regressions <- function(dt) {
    dt1 <- dt[period == 1]
    lapply(PROJ_VARS, function(pv) run_period1_spec(dt1, pv))
}

run_period1_spec <- function(dt1, proj_var) {
    fml <- as.formula(paste(
        "effort ~", proj_var, "+ ingroup + commonknow"
    ))
    feols(fml, data = dt1, cluster = ~session)
}

# =====
# Interaction regressions (e)
# =====
run_interaction_regressions <- function(dt) {
    ingroup_int <- run_interaction(dt, "ingroup")
    ck_int <- run_interaction(dt, "commonknow")
    c(ingroup_int, ck_int)
}

run_interaction <- function(dt, interact_var) {
    lapply(PROJ_VARS, function(pv) {
        fml <- as.formula(paste(
            "effort ~", pv, "*", interact_var, "| session + period_f"
        ))
        feols(fml, data = dt, cluster = ~session)
    })
}

# =====
# By-site regressions (f)
# =====
run_by_site_regressions <- function(dt) {
    sites <- levels(dt$run)
    site_models <- lapply(sites, function(s) {
        run_site_spec(dt[run == s], s)
    })
    names(site_models) <- paste0("run_", sites)
    site_models
}

run_site_spec <- function(dt_site, site_label) {
    fml <- effort ~ proj_cooperative + ingroup + commonknow | session + period_f
    feols(fml, data = dt_site, cluster = ~session)
}

# =====
# Group-level cross-section (g)
# =====
run_group_regressions <- function(dt) {
    models <- lapply(PROJ_VARS, function(pv) run_group_spec(dt, pv))
    multi <- run_group_multivariate(dt)
    c(models, list(multi))
}

run_group_spec <- function(dt, proj_var) {
    fml <- as.formula(paste(
        "mean_effort ~", proj_var, "+ ingroup + commonknow"
    ))
    feols(fml, data = dt, cluster = ~session)
}

run_group_multivariate <- function(dt) {
    rhs <- paste(PROJ_VARS, collapse = " + ")
    fml <- as.formula(paste(
        "mean_effort ~", rhs, "+ ingroup + commonknow"
    ))
    feols(fml, data = dt, cluster = ~session)
}

# =====
# LaTeX export
# =====
export_all_tables <- function(
    models_panel, models_period1, models_interact,
    models_site, models_group
) {
    export_comparison_table(models_panel, models_group, models_period1)
    export_detailed_table(models_panel, models_interact)
    export_period1_table(models_period1)
    export_interaction_table(models_interact)
    export_site_table(models_site)
    export_group_table(models_group)
}

export_comparison_table <- function(models_panel, models_group, models_p1) {
    # Spec B univariate coop is models_panel[[6]], multi is [[11]]
    # Group univariate coop is models_group[[1]], multi is [[6]]
    # Period-1 univariate coop is models_p1[[1]]
    selected <- list(
        models_panel[[6]], models_panel[[11]],
        models_group[[1]], models_group[[6]],
        models_p1[[1]]
    )
    headers <- c("Panel Coop", "Panel All", "Group Coop", "Group All", "P1 Coop")
    etable(
        selected, file = OUTPUT_TEX, tex = TRUE,
        fitstat = c("n", "r2", "ar2"),
        dict = build_var_dict(), headers = headers,
        title = "CCR External Validation: Effort Regressions"
    )
    cat("Exported:", OUTPUT_TEX, "\n")
}

export_detailed_table <- function(models_panel, models_interact) {
    # Panel: 5 spec_a + 5 spec_b + multi_b + 2 interactions (coop only)
    interact_coop <- list(models_interact[[1]], models_interact[[6]])
    all_models <- c(models_panel, interact_coop)
    headers <- c(
        rep("Spec A", 5), rep("Spec B", 5), "All B",
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

export_site_table <- function(models_site) {
    outfile <- sub("\\.tex$", "_by_site.tex", OUTPUT_TEX)
    etable(
        models_site, file = outfile, tex = TRUE,
        fitstat = c("n", "r2", "ar2"),
        dict = build_var_dict(),
        title = "CCR External Validation: By Site (Run)"
    )
    cat("Exported:", outfile, "\n")
}

export_group_table <- function(models) {
    outfile <- sub("\\.tex$", "_group.tex", OUTPUT_TEX)
    etable(
        models, file = outfile, tex = TRUE,
        fitstat = c("n", "r2", "ar2"),
        dict = build_var_dict(),
        title = "CCR External Validation: Group Cross-Section"
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
        commonknow = "Common Knowledge"
    )
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
