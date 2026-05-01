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
# This produces separate event-study coefficients for IF and AF,
# allowing heterogeneous treatment effects across treatments.
#
# Treatment coding: 1 = IF (Individual Feedback), 2 = AF (Aggregate Feedback).
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
OUTPUT_WALD_CSV <- file.path(OUTPUT_DIR, "issue_59_het_did_wald_pretrends.csv")

# Pre-period taus for joint Wald test (reference periods 0 and 999 are omitted by i()).
# tau=-6 is excluded from the joint test because it is a thinly-supported boundary bin.
PRE_TAUS <- -5:-1

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
    dt <- load_and_prepare_data(INPUT_CSV)
    validate_data(dt)
    models_info <- fit_all_models(dt)
    models <- lapply(models_info, `[[`, "model")
    print_summary_stats(dt)
    print_all_models(models[[1]], models[[2]], models[[3]], models[[4]])
    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
    export_latex_table(models, OUTPUT_TEX)
    cat("\nRegression table exported to:", OUTPUT_TEX, "\n")
    run_and_save_pretrend_walds(models_info, OUTPUT_WALD_CSV)
}

fit_all_models <- function(dt) {
    list(
        k20_main   = list(model = run_het_did_regression(dt, "20", "did_sample_20"),        threshold = "20"),
        k20_robust = list(model = run_het_did_regression(dt, "20", "did_sample_robust_20"), threshold = "20"),
        k5_main    = list(model = run_het_did_regression(dt, "5",  "did_sample_5"),         threshold = "5"),
        k5_robust  = list(model = run_het_did_regression(dt, "5",  "did_sample_robust_5"),  threshold = "5")
    )
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
    cat("IF:", sum(dt$treatment == "1"), "\n")
    cat("AF:", sum(dt$treatment == "2"), "\n\n")
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
            trt_label <- ifelse(trt == "t1", "IF", "AF")
            labs <- sprintf("%s Treated $\\times$ $\\tau = %d$", trt_label, taus)
            dict <- c(dict, setNames(labs, nms))
        }
    }
    dict <- c(dict, "treatment2" = "AF (vs IF)")
    return(dict)
}

# =====
# Joint pre-trend Wald tests per treatment arm
# =====
pretrend_coef_pattern <- function(threshold, arm) {
    # Matches e.g. "tau_20::-6:suckered_t1_20" for tau in -6..-1
    taus <- paste(PRE_TAUS, collapse = "|")
    sprintf("^tau_%s::(%s):suckered_%s_%s$", threshold, taus, arm, threshold)
}

wald_pretrend_one_arm <- function(model, threshold, arm) {
    # fixest::wald() uses the same vcov the model was fit with (clustered here).
    # Coefficient names use lowercase arm ("t1"/"t2"); keep that case in the regex.
    pattern <- pretrend_coef_pattern(threshold, arm)
    matched <- grep(pattern, names(coef(model)), value = TRUE)
    # Empty match would make fixest::wald return a bare NA (atomic), so guard here.
    if (length(matched) == 0L) {
        return(list(chisq = NA_real_, df = 0L, p_value = NA_real_))
    }
    res <- fixest::wald(model, keep = pattern, print = FALSE)
    list(chisq = unname(res[["stat"]]), df = unname(res[["df1"]]), p_value = unname(res[["p"]]))
}

build_wald_row <- function(model_label, arm, threshold, model) {
    res <- wald_pretrend_one_arm(model, threshold, arm)
    data.table(
        model = model_label,
        arm = toupper(arm),
        chisq = res$chisq,
        df = res$df,
        p_value = res$p_value
    )
}

run_and_save_pretrend_walds <- function(models_info, filepath) {
    cat("\n=== Joint Pre-trend Wald Tests (clustered vcov) ===\n")
    rows <- list()
    for (label in names(models_info)) {
        info <- models_info[[label]]
        for (arm in c("t1", "t2")) {
            row <- build_wald_row(label, arm, info$threshold, info$model)
            rows[[length(rows) + 1L]] <- row
            cat(sprintf(
                "%s | %s: chisq=%.4f, df=%d, p=%.4f\n",
                label, toupper(arm), row$chisq, row$df, row$p_value
            ))
        }
    }
    out <- rbindlist(rows)
    fwrite(out, filepath)
    cat("Pre-trend Wald results saved to:", filepath, "\n")
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
