# Purpose: Estimate lying contagion — does a participant's lying in round t
#   respond to OTHER group members' lying, heterogeneous by treatment?
#   Pooled logit specifications (A: lag; B: cumulative), group-clustered.
#   Reports joint Wald test of TOTAL IF effect
#   (H0: beta_main + beta_{main x IF} = 0) at the bottom.
# Author: Claude Code
# Date: 2026-04-20

library(data.table)
library(fixest)

# FILE PATHS
PANEL_CSV <- "datastore/derived/issue_72_panel.csv"
OUTPUT_DIR <- "output/tables"
OUTPUT_TEX <- file.path(OUTPUT_DIR, "issue_72_lying_contagion.tex")

VAR_DICT <- c(
    group_lied_lag         = "Group Lied (t-1)",
    self_lied_lag          = "Self Lied (t-1)",
    any_group_lied_prior   = "Any Group Lied Prior",
    any_self_lied_prior    = "Any Self Lied Prior",
    treatment_f            = "Treatment",
    segment                = "Segment",
    round                  = "Round"
)

MODEL_HEADERS <- c("Logit A", "Logit B")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
    dt <- load_panel(PANEL_CSV)
    cat(sprintf("Loaded panel: %d obs, %d individuals, %d sessions\n",
                nrow(dt), uniqueN(dt$label_session), uniqueN(dt$session_code)))

    models <- estimate_all_models(dt)
    tests <- compute_joint_tests(models)

    print_model_summaries(models)
    print_test_results(tests)

    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
    export_table(models, tests, OUTPUT_TEX)
    cat("Table exported to:", OUTPUT_TEX, "\n")
}

# =====
# Data loading
# =====
REQUIRED_PANEL_COLS <- c("session_code", "treatment", "segment", "round",
                         "cluster_group", "label_session", "lied",
                         "group_lied_lag", "self_lied_lag",
                         "any_group_lied_prior", "any_self_lied_prior")

load_panel <- function(path) {
    if (!file.exists(path)) {
        stop(sprintf(
            "Panel CSV not found: %s. Run analysis/derived/build_issue_72_panel.py first.",
            path))
    }
    dt <- as.data.table(read.csv(path))
    validate_panel(dt, path)
    # AF (treatment 2) as reference → interaction reports IF contrast.
    stopifnot("treatment must be in {1, 2, NA}" =
                  all(dt$treatment %in% c(1, 2, NA)))
    dt[, treatment_f := relevel(
        factor(ifelse(treatment == 1, "IF", "AF")), ref = "AF")]
    return(dt)
}

validate_panel <- function(dt, path) {
    missing_cols <- setdiff(REQUIRED_PANEL_COLS, names(dt))
    if (length(missing_cols) > 0) {
        stop(sprintf("Panel missing required columns: %s. Available: %s",
                     paste(missing_cols, collapse = ", "),
                     paste(names(dt), collapse = ", ")))
    }
    if (nrow(dt) == 0) {
        stop(sprintf("Panel has zero rows: %s", path))
    }
}

# =====
# Regression models (pooled logit: no individual FE)
# =====
estimate_all_models <- function(dt) {
    list(
        m5 = pooled_logit_version_a(dt),
        m6 = pooled_logit_version_b(dt)
    )
}

pooled_logit_version_a <- function(dt) {
    feglm(
        lied ~ group_lied_lag + self_lied_lag + i(treatment_f, ref = "AF") +
            i(treatment_f, ref = "AF"):group_lied_lag |
            segment + round,
        data = dt,
        family = binomial(link = "logit"),
        cluster = ~cluster_group
    )
}

pooled_logit_version_b <- function(dt) {
    feglm(
        lied ~ any_group_lied_prior + any_self_lied_prior +
            i(treatment_f, ref = "AF") +
            i(treatment_f, ref = "AF"):any_group_lied_prior |
            segment + round,
        data = dt,
        family = binomial(link = "logit"),
        cluster = ~cluster_group
    )
}

# =====
# Joint Wald test: H0: beta_main + beta_interaction = 0
# Tests whether total group-lying contagion effect under IF is zero.
# Uses clustered vcov stored in the fitted model.
# =====
compute_joint_tests <- function(models) {
    list(
        m5 = joint_test(models$m5,
                        main_var = "group_lied_lag",
                        int_var  = "group_lied_lag:treatment_f::IF"),
        m6 = joint_test(models$m6,
                        main_var = "any_group_lied_prior",
                        int_var  = "any_group_lied_prior:treatment_f::IF")
    )
}

joint_test <- function(model, main_var, int_var) {
    b <- coef(model)
    V <- vcov(model)
    check_coefs_present(b, c(main_var, int_var))
    k <- rep(0, length(b))
    k[names(b) == main_var] <- 1
    k[names(b) == int_var]  <- 1
    est <- sum(k * b)
    var_est <- as.numeric(t(k) %*% V %*% k)
    check_variance(var_est, main_var, int_var)
    chi2 <- (est^2) / var_est
    list(
        estimate = est,
        se       = sqrt(var_est),
        chi2     = chi2,
        pvalue   = pchisq(chi2, df = 1, lower.tail = FALSE)
    )
}

check_coefs_present <- function(b, requested_vars) {
    missing_vars <- setdiff(requested_vars, names(b))
    if (length(missing_vars) > 0) {
        stop(sprintf("joint_test: missing coefficients: %s. Available: %s",
                     paste(missing_vars, collapse = ", "),
                     paste(names(b), collapse = ", ")))
    }
}

check_variance <- function(var_est, main_var, int_var) {
    if (!is.finite(var_est) || var_est <= 0) {
        stop(sprintf(paste0("joint_test: non-positive/NA variance (%.6g) ",
                            "for contrast (%s + %s). ",
                            "Possible rank deficiency or singular vcov."),
                     var_est, main_var, int_var))
    }
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

print_test_results <- function(tests) {
    cat("\n===== Joint Wald Tests (H0: beta_main + beta_int = 0) =====\n")
    for (nm in names(tests)) {
        t <- tests[[nm]]
        cat(sprintf("  %s: est=%.4f  SE=%.4f  chi2=%.3f  p=%.4f\n",
                    nm, t$estimate, t$se, t$chi2, t$pvalue))
    }
}

# =====
# LaTeX export
# =====
sig_stars <- function(p) {
    if (p < 0.01) return("$^{***}$")
    if (p < 0.05) return("$^{**}$")
    if (p < 0.10) return("$^{*}$")
    ""
}

format_chi2_row <- function(tests) {
    sapply(tests, function(t)
        sprintf("%.3f%s", t$chi2, sig_stars(t$pvalue)))
}

format_pvalue_row <- function(tests) {
    sapply(tests, function(t) sprintf("%.3f", t$pvalue))
}

export_table <- function(models, tests, filepath) {
    etable(
        models$m5, models$m6,
        file = filepath,
        replace = TRUE,
        tex = TRUE,
        fitstat = c("n"),
        dict = VAR_DICT,
        headers = MODEL_HEADERS,
        title = "Lying Contagion: Pooled Logit by Treatment",
        se.below = TRUE,
        extralines = list(
            `__Wald $\\chi^2$ ($\\beta_{\\text{main}} + \\beta_{\\times \\text{IF}} = 0$)` =
                format_chi2_row(tests),
            `__\\quad $p$-value` = format_pvalue_row(tests)
        )
    )
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
