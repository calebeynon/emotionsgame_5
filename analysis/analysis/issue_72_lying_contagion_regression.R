# Purpose: Estimate lying contagion — does a participant's lying in round t
#   respond to OTHER group members' lying, heterogeneous by treatment?
#   Six models: two specifications (lag / cumulative) x two clusterings
#   (group / session) x two link functions (LPM / logit, group-clustered only).
# Author: Claude Code
# Date: 2026-04-19

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
    "LPM A (grp)", "LPM A (ses)",
    "LPM B (grp)", "LPM B (ses)",
    "Logit A (grp)", "Logit B (grp)"
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
        m1 = lpm_version_a(dt, cluster_var = ~cluster_group),
        m2 = lpm_version_a(dt, cluster_var = ~session_code),
        m3 = lpm_version_b(dt, cluster_var = ~cluster_group),
        m4 = lpm_version_b(dt, cluster_var = ~session_code),
        m5 = logit_version_a(dt),
        m6 = logit_version_b(dt)
    )
}

# Version A: one-round lag regressors.
lpm_version_a <- function(dt, cluster_var) {
    feols(
        lied ~ group_lied_lag + self_lied_lag + i(treatment_f, ref = 2) +
            i(treatment_f, ref = 2):group_lied_lag |
            session_code + segment + round + label_session,
        cluster = cluster_var,
        data = dt
    )
}

# Version B: cumulative "any prior round" regressors.
lpm_version_b <- function(dt, cluster_var) {
    feols(
        lied ~ any_group_lied_prior + any_self_lied_prior +
            i(treatment_f, ref = 2) +
            i(treatment_f, ref = 2):any_group_lied_prior |
            session_code + segment + round + label_session,
        cluster = cluster_var,
        data = dt
    )
}

logit_version_a <- function(dt) {
    feglm(
        lied ~ group_lied_lag + self_lied_lag + i(treatment_f, ref = 2) +
            i(treatment_f, ref = 2):group_lied_lag |
            session_code + segment + round + label_session,
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
            session_code + segment + round + label_session,
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
