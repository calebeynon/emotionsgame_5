# Purpose: Estimate lying contagion — does a participant's lying in round t
#   respond to OTHER group members' lying, heterogeneous by treatment?
#   Six models: two specifications (lag / cumulative) x three estimators
#   (LPM, FE Logit, Pooled Logit). All group-clustered.
# Author: Claude Code
# Date: 2026-04-20

library(data.table)
library(fixest)

# FILE PATHS
PANEL_CSV <- "datastore/derived/issue_72_panel.csv"
OUTPUT_DIR <- "output/tables"
OUTPUT_TEX <- file.path(OUTPUT_DIR, "issue_72_lying_contagion.tex")

# Variable label dictionary for etable() output.
VAR_DICT <- c(
    group_lied_lag         = "Group Lied (t-1)",
    self_lied_lag          = "Self Lied (t-1)",
    any_group_lied_prior   = "Any Group Lied Prior",
    any_self_lied_prior    = "Any Self Lied Prior",
    treatment_f            = "Treatment",
    session_code           = "Session",
    segment                = "Segment",
    round                  = "Round",
    label_session          = "Individual"
)

MODEL_HEADERS <- c(
    "LPM A", "LPM B",
    "FE Logit A", "FE Logit B",
    "Pooled Logit A", "Pooled Logit B"
)

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
    dt <- load_panel(PANEL_CSV)
    cat(sprintf("Loaded panel: %d obs, %d individuals, %d sessions\n",
                nrow(dt), uniqueN(dt$label_session), uniqueN(dt$session_code)))

    models <- estimate_all_models(dt)
    print_model_summaries(models)

    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
    export_table(models, OUTPUT_TEX)
    cat("Table exported to:", OUTPUT_TEX, "\n")
}

# =====
# Data loading
# =====
load_panel <- function(path) {
    dt <- as.data.table(read.csv(path))
    # Factor treatment with level "2" as reference so i(treatment_f, ref=2)
    # reports the Treatment 1 vs Treatment 2 contrast.
    dt[, treatment_f := relevel(factor(treatment), ref = "2")]
    return(dt)
}

# =====
# Regression models
# =====
estimate_all_models <- function(dt) {
    list(
        m1 = lpm_version_a(dt),
        m2 = lpm_version_b(dt),
        m3 = logit_version_a(dt),
        m4 = logit_version_b(dt),
        m5 = pooled_logit_version_a(dt),
        m6 = pooled_logit_version_b(dt)
    )
}

# Version A: one-round lag regressors. LPM with individual FE.
lpm_version_a <- function(dt) {
    feols(
        lied ~ group_lied_lag + self_lied_lag + i(treatment_f, ref = 2) +
            i(treatment_f, ref = 2):group_lied_lag |
            segment + round + label_session,
        cluster = ~cluster_group,
        data = dt
    )
}

# Version B: cumulative "any prior round" regressors. LPM with individual FE.
lpm_version_b <- function(dt) {
    feols(
        lied ~ any_group_lied_prior + any_self_lied_prior +
            i(treatment_f, ref = 2) +
            i(treatment_f, ref = 2):any_group_lied_prior |
            segment + round + label_session,
        cluster = ~cluster_group,
        data = dt
    )
}

# FE Logit (individual FE): drops never-liars, retains only within-individual variation.
logit_version_a <- function(dt) {
    feglm(
        lied ~ group_lied_lag + self_lied_lag + i(treatment_f, ref = 2) +
            i(treatment_f, ref = 2):group_lied_lag |
            segment + round + label_session,
        data = dt,
        family = binomial(link = "logit"),
        cluster = ~cluster_group
    )
}

logit_version_b <- function(dt) {
    feglm(
        lied ~ any_group_lied_prior + any_self_lied_prior +
            i(treatment_f, ref = 2) +
            i(treatment_f, ref = 2):any_group_lied_prior |
            segment + round + label_session,
        data = dt,
        family = binomial(link = "logit"),
        cluster = ~cluster_group
    )
}

# Pooled logit: no individual FE, no session FE. Keeps all 160 individuals
# (N = 2,720) and identifies the Treatment main effect. Clustered at group level.
pooled_logit_version_a <- function(dt) {
    feglm(
        lied ~ group_lied_lag + self_lied_lag + i(treatment_f, ref = 2) +
            i(treatment_f, ref = 2):group_lied_lag |
            segment + round,
        data = dt,
        family = binomial(link = "logit"),
        cluster = ~cluster_group
    )
}

pooled_logit_version_b <- function(dt) {
    feglm(
        lied ~ any_group_lied_prior + any_self_lied_prior +
            i(treatment_f, ref = 2) +
            i(treatment_f, ref = 2):any_group_lied_prior |
            segment + round,
        data = dt,
        family = binomial(link = "logit"),
        cluster = ~cluster_group
    )
}

# =====
# Diagnostics
# =====
print_model_summaries <- function(models) {
    for (nm in names(models)) {
        cat("\n===== Model:", nm, "=====\n")
        print(summary(models[[nm]]))
    }
}

# =====
# LaTeX export
# =====
export_table <- function(models, filepath) {
    etable(
        models$m1, models$m2, models$m3, models$m4, models$m5, models$m6,
        file = filepath,
        replace = TRUE,
        tex = TRUE,
        fitstat = c("n"),
        dict = VAR_DICT,
        headers = MODEL_HEADERS,
        title = "Lying Contagion: Response to Group Lying by Treatment",
        se.below = TRUE
    )
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
