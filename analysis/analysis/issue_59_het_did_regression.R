# Purpose: Heterogeneous treatment diff-in-diff event study regression of
#          contribution behavior after being suckered, with separate coefficients
#          by treatment group.
# Author: Claude Code
# Date: 2026-04-10
#
# Model: contribution ~ i(tau, suckered_t1, ref = c(0, 999))
#                      + i(tau, suckered_t2, ref = c(0, 999)) + treatment
#                      | round + segment
#
# This produces separate event-study coefficients for Treatment 1 and Treatment 2,
# allowing heterogeneous treatment effects across treatments.
# Treatment-specific suckered indicators are constructed by interacting got_suckered
# with treatment group dummies.
#
# Clustering: session_code + segment + group (concatenated as cluster_id)
#
# Four models (two thresholds x two sample definitions):
#   < 20 (tau_20): groupmate broke promise by contributing < 20 after promising
#   < 5  (tau_5):  groupmate broke promise by contributing < 5 after promising
#   Main: controls are non-suckered players
#   Robust: controls restricted to always-cooperators (never broke a promise)

# nolint start
library(data.table)
library(fixest)

# FILE PATHS
INPUT_CSV <- "datastore/derived/issue_20_did_panel.csv"
OUTPUT_DIR <- "output/tables"
OUTPUT_TEX <- file.path(OUTPUT_DIR, "issue_59_het_did_contribution.tex")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
    dt <- load_and_prepare_data(INPUT_CSV)
    validate_data(dt)

    model_20_main   <- run_het_did_regression(dt, "20", "did_sample_20")
    model_20_robust <- run_het_did_regression(dt, "20", "did_sample_robust_20")
    model_5_main    <- run_het_did_regression(dt, "5", "did_sample_5")
    model_5_robust  <- run_het_did_regression(dt, "5", "did_sample_robust_5")

    print_summary_stats(dt)
    print_all_models(model_20_main, model_20_robust, model_5_main, model_5_robust)

    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
    models <- list(model_20_main, model_20_robust, model_5_main, model_5_robust)
    export_latex_table(models, OUTPUT_TEX)
    cat("\nRegression table exported to:", OUTPUT_TEX, "\n")
}

# =====
# Data loading and preparation
# =====
load_and_prepare_data <- function(filepath) {
    dt <- fread(filepath)
    convert_bool_cols(dt)
    prepare_tau_cols(dt)
    dt[, treatment := factor(treatment)]
    return(dt)
}

convert_bool_cols <- function(dt) {
    bool_cols <- c(
        "got_suckered_20", "got_suckered_5",
        "did_sample_20", "did_sample_5",
        "did_sample_robust_20", "did_sample_robust_5"
    )
    for (col in bool_cols) {
        dt[, (col) := as.integer(as.logical(get(col)))]
    }
}

prepare_tau_cols <- function(dt) {
    # Sentinel value (999) for control players' NA tau, excluded via ref
    dt[is.na(tau_20), tau_20 := 999]
    dt[is.na(tau_5), tau_5 := 999]
    dt[, tau_20 := as.integer(tau_20)]
    dt[, tau_5 := as.integer(tau_5)]
}

# =====
# Data validation
# =====
validate_data <- function(dt) {
    required_cols <- c(
        "contribution", "got_suckered_20", "got_suckered_5",
        "tau_20", "tau_5", "did_sample_20", "did_sample_5",
        "did_sample_robust_20", "did_sample_robust_5",
        "treatment", "round", "segment", "cluster_id"
    )
    missing <- setdiff(required_cols, names(dt))
    if (length(missing) > 0) {
        stop("Missing required columns: ", paste(missing, collapse = ", "))
    }
}

# =====
# Treatment-specific suckered indicators
# =====
create_treatment_indicators <- function(dt_sub, threshold) {
    suckered_col <- paste0("got_suckered_", threshold)
    t1_col <- paste0("suckered_t1_", threshold)
    t2_col <- paste0("suckered_t2_", threshold)
    dt_sub[, (t1_col) := as.integer(get(suckered_col) == 1 & treatment == "1")]
    dt_sub[, (t2_col) := as.integer(get(suckered_col) == 1 & treatment == "2")]
}

# =====
# Heterogeneous DiD regression estimation
# =====
run_het_did_regression <- function(dt, threshold, sample_col) {
    tau_col <- paste0("tau_", threshold)
    t1_col <- paste0("suckered_t1_", threshold)
    t2_col <- paste0("suckered_t2_", threshold)
    dt_sub <- dt[get(sample_col) == 1]
    create_treatment_indicators(dt_sub, threshold)

    formula_str <- sprintf(
        "contribution ~ i(%s, %s, ref = c(0, 999)) + i(%s, %s, ref = c(0, 999)) + treatment | round + segment",
        tau_col, t1_col, tau_col, t2_col
    )

    feols(as.formula(formula_str), data = dt_sub, cluster = ~cluster_id)
}

# =====
# Summary statistics
# =====
print_summary_stats <- function(dt) {
    cat("=== Summary Statistics (by Treatment) ===\n")
    cat("Total observations:", nrow(dt), "\n")
    cat("Treatment 1:", sum(dt$treatment == "1"), "\n")
    cat("Treatment 2:", sum(dt$treatment == "2"), "\n\n")
    for (thresh in c("20", "5")) {
        print_threshold_stats(dt, thresh, paste0("did_sample_", thresh), "Main")
        print_threshold_stats(dt, thresh, paste0("did_sample_robust_", thresh), "Robust")
    }
}

print_threshold_stats <- function(dt, thresh, sample_col, label) {
    suckered_col <- paste0("got_suckered_", thresh)
    for (trt in c("1", "2")) {
        mask <- dt[[sample_col]] == 1 & dt$treatment == trt
        n_sample <- sum(mask)
        n_treated <- sum(mask & dt[[suckered_col]] == 1)
        n_control <- sum(mask & dt[[suckered_col]] == 0)
        cat(sprintf(
            "Threshold %s (%s), Treatment %s: %d obs (%d treated, %d control)\n",
            thresh, label, trt, n_sample, n_treated, n_control
        ))
    }
}

# =====
# LaTeX output
# =====
print_all_models <- function(m20_main, m20_robust, m5_main, m5_robust) {
    labels <- c("< 20 (Main)", "< 20 (Robust)", "< 5 (Main)", "< 5 (Robust)")
    models <- list(m20_main, m20_robust, m5_main, m5_robust)
    for (i in seq_along(models)) {
        cat(sprintf("\n--- %s Model ---\n", labels[i]))
        print(summary(models[[i]]))
    }
}

export_latex_table <- function(models, filepath) {
    coef_names <- build_coef_dict()
    etable(
        models[[1]], models[[2]], models[[3]], models[[4]],
        file = filepath,
        tex = TRUE,
        fitstat = c("n", "r2"),
        dict = coef_names,
        headers = list(
            "Threshold" = c("$<$ 20", "$<$ 20", "$<$ 5", "$<$ 5"),
            "Sample" = c("Main", "Robust", "Main", "Robust")
        ),
        notes = "Empty cells indicate no observations at that event time for the treatment group."
    )
}

build_coef_dict <- function() {
    taus <- c(-6:-1, 1:5)
    dict <- c()
    for (thresh in c("20", "5")) {
        for (trt in c("t1", "t2")) {
            col <- paste0("suckered_", trt, "_", thresh)
            nms <- sprintf("tau_%s::%d:%s", thresh, taus, col)
            trt_label <- ifelse(trt == "t1", "T1", "T2")
            labs <- sprintf("%s Treated $\\times$ $\\tau = %d$", trt_label, taus)
            dict <- c(dict, setNames(labs, nms))
        }
    }
    dict <- c(dict, "treatment2" = "Treatment 2")
    return(dict)
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
