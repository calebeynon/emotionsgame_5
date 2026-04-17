# Purpose: Dump coefficient + Wald results for pytest validation (issue #57)
# Author: Claude Code (test-writer)
# Date: 2026-04-16
# nolint start
library(data.table)
library(plm)

# FILE PATHS
INPUT_CSV <- "datastore/derived/dynamic_regression_panel.csv"
OUTPUT_DIR <- "output/tables"
COEFS_CSV <- file.path(OUTPUT_DIR, "dynamic_regression_coefs.csv")
WALD_CSV <- file.path(OUTPUT_DIR, "dynamic_regression_wald.csv")

MODEL_LABELS <- c("T1_baseline", "T1_chat", "T1_chatfacial",
                  "T2_baseline", "T2_chat", "T2_chatfacial")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
    TESTING <<- TRUE
    source("analysis/dynamic_regression.R", local = FALSE)

    dt <- load_and_prepare_data(INPUT_CSV)
    formulas <- build_formulas()

    panels_t1 <- list(prepare_panel(dt[treatment == 1]),
                      prepare_panel(dt[treatment == 1]),
                      prepare_panel(dt[treatment == 1 & !is.na(emotion_valence)]))
    panels_t2 <- list(prepare_panel(dt[treatment == 2]),
                      prepare_panel(dt[treatment == 2]),
                      prepare_panel(dt[treatment == 2 & !is.na(emotion_valence)]))

    models <- estimate_labeled_models(panels_t1, panels_t2, formulas)
    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
    write_coefs_csv(models, COEFS_CSV)
    write_wald_csv(models, WALD_CSV)
    cat("Wrote:", COEFS_CSV, "and", WALD_CSV, "\n")
}

# =====
# Estimate 6 models and return named list (T1_baseline, ..., T2_chatfacial)
# =====
estimate_labeled_models <- function(panels_t1, panels_t2, formulas) {
    models <- list()
    for (i in seq_along(formulas)) {
        models[[MODEL_LABELS[i]]]     <- run_arellano_bond(panels_t1[[i]], formulas[[i]])
        models[[MODEL_LABELS[i + 3]]] <- run_arellano_bond(panels_t2[[i]], formulas[[i]])
    }
    return(models)
}

# =====
# Write coefficients CSV: model, term, coef, se, pvalue
# =====
write_coefs_csv <- function(models, filepath) {
    rows <- list()
    for (label in names(models)) {
        s <- summary(models[[label]], robust = TRUE)
        coef_mat <- s$coefficients
        for (term in rownames(coef_mat)) {
            rows[[length(rows) + 1]] <- data.table(
                model = label,
                term = term,
                coef = coef_mat[term, 1],
                se = coef_mat[term, 2],
                pvalue = coef_mat[term, 4]
            )
        }
    }
    fwrite(rbindlist(rows), filepath)
}

# =====
# Write Wald tests CSV: model, test_name, pvalue
# =====
write_wald_csv <- function(models, filepath) {
    dev_vars <- c("contmore_L1", "contless_L1")
    rd_vars <- c("round1", "round2")
    rows <- list()
    for (label in names(models)) {
        rows[[length(rows) + 1]] <- data.table(
            model = label,
            test_name = "pos_plus_neg",
            pvalue = wald_test_pvalue(models[[label]], dev_vars)
        )
        rows[[length(rows) + 1]] <- data.table(
            model = label,
            test_name = "R1_plus_R2",
            pvalue = wald_test_pvalue(models[[label]], rd_vars)
        )
    }
    fwrite(rbindlist(rows), filepath)
}

# %%
main()
