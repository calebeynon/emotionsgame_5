# Purpose: Dynamic panel regression (Arellano-Bond GMM) of contribution dynamics
# Author: Claude Code
# Date: 2026-04-09
#
# Translates analysis/analysis/dynamic_regression.do to R
# Model: xtabond equivalent with 2 lags of dependent variable, two-step GMM
# Separate estimations for Treatment 1 and Treatment 2
# Instruments: lags 2-4 of contribution (maxldep=4, maxlags=4)

# nolint start
library(data.table)
library(plm)
library(texreg)

# FILE PATHS
INPUT_CSV <- "datastore/derived/contributions.csv"
OUTPUT_DIR <- "output/tables"
OUTPUT_TEX <- file.path(OUTPUT_DIR, "dynamic_regression.tex")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
    dt <- load_and_prepare_data(INPUT_CSV)

    pdata_t1 <- prepare_panel(dt[treatment == 1])
    pdata_t2 <- prepare_panel(dt[treatment == 2])

    model_t1 <- run_arellano_bond(pdata_t1)
    model_t2 <- run_arellano_bond(pdata_t2)

    print_diagnostics(model_t1, "Treatment 1")
    print_diagnostics(model_t2, "Treatment 2")

    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
    export_latex_table(model_t1, model_t2, OUTPUT_TEX)
    cat("\nTable exported to:", OUTPUT_TEX, "\n")
}

# =====
# Data loading and variable construction
# =====
load_and_prepare_data <- function(filepath) {
    dt <- as.data.table(read.csv(filepath))
    dt <- derive_deviation_variables(dt)
    dt <- create_panel_variables(dt)
    return(dt)
}

# =====
# Derive others' contributions and deviation measures
# =====
derive_deviation_variables <- function(dt) {
    # payoff = 25 - contribution + (contribution + othercont) * 0.4
    dt[, othercont := (payoff - 25 + 0.6 * contribution) / 0.4]
    dt[, othercontaverage := othercont / 3]  # corrected from /1.3

    dt[, morethanaverage := as.integer(contribution > othercontaverage)]
    dt[, lessthanaverage := as.integer(contribution < othercontaverage)]

    dt[, diffcont := contribution - othercontaverage]
    dt[, contmore := diffcont * morethanaverage]
    dt[, contless := -diffcont * lessthanaverage]
    return(dt)
}

# =====
# Create panel structure: period, subject_id, lags, round dummies
# =====
create_panel_variables <- function(dt) {
    dt[, segmentnumber := match(segment, paste0("supergame", 1:5))]

    # Linearize period across supergames (matching Stata code)
    period_offsets <- c(0L, 3L, 7L, 10L, 17L)
    dt[, period := as.integer(round) + period_offsets[segmentnumber]]

    # Unique subject ID across sessions
    dt[, sessionnumber := match(session_code, unique(session_code))]
    dt[, subject_id := sessionnumber * 100L + participant_id]

    dt <- create_lag_and_dummy_vars(dt)
    return(dt)
}

# =====
# Create lagged deviations and round dummies
# =====
create_lag_and_dummy_vars <- function(dt) {
    setorder(dt, subject_id, period)
    dt[, contmore_L1 := shift(contmore, 1), by = subject_id]
    dt[period == 1, contmore_L1 := NA]
    dt[, contless_L1 := shift(contless, 1), by = subject_id]
    dt[period == 1, contless_L1 := NA]
    for (r in 1:7) dt[, paste0("round", r) := as.integer(round == r)]
    return(dt)
}

# =====
# Panel data preparation
# =====
prepare_panel <- function(dt) {
    pdata.frame(as.data.frame(dt), index = c("subject_id", "period"))
}

# =====
# Arellano-Bond estimation
# =====
run_arellano_bond <- function(pdata) {
    pgmm(
        contribution ~ lag(contribution, 1:2) +
            contmore_L1 + contless_L1 +
            round1 + round2 + round3 + round4 + round5 +
            segmentnumber |
            lag(contribution, 2:4),
        data = pdata,
        effect = "individual",
        model = "twosteps",
        transformation = "d"
    )
}

# =====
# Print diagnostic tests
# =====
print_diagnostics <- function(model, label) {
    cat("\n===", label, "===\n")
    print(summary(model, robust = TRUE))
}

# =====
# LaTeX table export
# =====
export_latex_table <- function(m1, m2, filepath) {
    s1 <- summary(m1, robust = TRUE)
    s2 <- summary(m2, robust = TRUE)

    coef_names <- build_coef_names(m1)
    gof_rows <- build_gof_rows(m1, m2, s1, s2)

    tex_output <- texreg(
        list(m1, m2),
        custom.model.names = c("Treatment 1", "Treatment 2"),
        custom.coef.names = coef_names,
        override.se = list(s1$coefficients[, 2], s2$coefficients[, 2]),
        override.pvalues = list(s1$coefficients[, 4], s2$coefficients[, 4]),
        stars = c(0.01, 0.05, 0.1),
        table = FALSE, booktabs = TRUE, use.packages = FALSE, digits = 3,
        custom.gof.rows = gof_rows
    )

    writeLines(clean_tex_gof(tex_output), filepath)
}

# =====
# Build coefficient display names
# =====
build_coef_names <- function(model) {
    c(
        "Contribution$_{t-1}$", "Contribution$_{t-2}$",
        "Positive Deviation$_{t-1}$", "Negative Deviation$_{t-1}$",
        "Round 1", "Round 2", "Round 3", "Round 4", "Round 5",
        "Segment"
    )
}

# =====
# Wald test for linear hypothesis H0: Rβ = 0 using robust vcov
# =====
wald_test_pvalue <- function(robust_summary, coef_names) {
    beta <- robust_summary$coefficients[, 1]
    V <- robust_summary$vcov
    idx <- match(coef_names, names(beta))
    r <- beta[idx]
    W <- sum(r)^2 / sum(V[idx, idx])
    pchisq(W, df = 1, lower.tail = FALSE)
}

# =====
# Build custom GOF rows (observations, AR tests, Sargan, Wald tests)
# =====
build_gof_rows <- function(m1, m2, s1, s2) {
    # Wald test: positive + negative deviation = 0 (symmetry)
    dev_vars <- c("contmore_L1", "contless_L1")
    dev_p <- c(wald_test_pvalue(s1, dev_vars), wald_test_pvalue(s2, dev_vars))

    # Wald test: round1 + round2 = 0
    rd_vars <- c("round1", "round2")
    rd_p <- c(wald_test_pvalue(s1, rd_vars), wald_test_pvalue(s2, rd_vars))

    list(
        "Observations" = c(sum(sapply(m1$residuals, length)),
                          sum(sapply(m2$residuals, length))),
        "AR(1) p-value" = c(s1$m1$p.value, s2$m1$p.value),
        "AR(2) p-value" = c(s1$m2$p.value, s2$m2$p.value),
        "Sargan p-value" = c(s1$sargan$p.value, s2$sargan$p.value),
        "$\\beta_{\\text{pos}} + \\beta_{\\text{neg}} = 0$ (p)" = dev_p,
        "$\\beta_{R1} + \\beta_{R2} = 0$ (p)" = rd_p
    )
}

# =====
# Remove default texreg GOF rows, keep only custom ones
# =====
clean_tex_gof <- function(tex_output) {
    lines <- strsplit(tex_output, "\n")[[1]]
    drop <- c("^n ", "^T ", "^Num\\.", "^Sargan Test:", "^Wald Test")
    keep <- !grepl(paste(drop, collapse = "|"), trimws(lines))
    paste(lines[keep], collapse = "\n")
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
