# Purpose: Dump coefficient + Wald results for pytest validation
# Author: Claude Code
# Date: 2026-04-19 (issue #68: 4 baseline + 12 extended models)
# nolint start
library(data.table)
library(plm)

# FILE PATHS
INPUT_CSV <- "datastore/derived/dynamic_regression_panel.csv"
OUTPUT_DIR <- "output/tables"
COEFS_CSV <- file.path(OUTPUT_DIR, "dynamic_regression_coefs.csv")
WALD_CSV <- file.path(OUTPUT_DIR, "dynamic_regression_wald.csv")

main <- function() {
    TESTING <<- TRUE
    source("analysis/dynamic_regression.R", local = FALSE)

    dt <- load_and_prepare_data(INPUT_CSV)
    panels <- build_all_panels(dt)
    formulas <- build_formulas()
    models <- c(fit_baseline_models(panels, formulas),
                fit_extended_models(panels, formulas))

    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
    write_coefs_csv(models, COEFS_CSV)
    write_wald_csv(models, WALD_CSV)
    cat("Wrote:", COEFS_CSV, "and", WALD_CSV, "\n")
}

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

WALD_PAIRS <- list(
    pos_plus_neg = c("contmore_L1", "contless_L1"),
    max_pos_plus_neg = c("contmoremax_L1", "contlessmax_L1"),
    med_pos_plus_neg = c("contmoremed_L1", "contlessmed_L1"),
    min_pos_plus_neg = c("contmoremin_L1", "contlessmin_L1")
)

write_wald_csv <- function(models, filepath) {
    rows <- list()
    for (label in names(models)) {
        for (test_name in names(WALD_PAIRS)) {
            row <- wald_row_for(models, label, test_name)
            if (!is.null(row)) rows[[length(rows) + 1]] <- row
        }
    }
    fwrite(rbindlist(rows), filepath)
}

wald_row_for <- function(models, label, test_name) {
    pair <- WALD_PAIRS[[test_name]]
    present <- pair %in% names(coef(models[[label]]))
    # Warn on partial presence only; both-absent means the pair belongs to a
    # different family (mean vs min/med/max) and is expected to skip silently.
    if (any(present) && !all(present)) {
        warning(sprintf(
            "Wald test '%s' for model '%s' partially present: %s",
            test_name, label, paste(pair, collapse = "+")
        ))
    }
    if (!all(present)) return(NULL)
    data.table(
        model = label,
        test_name = test_name,
        pvalue = wald_test_pvalue(models[[label]], pair)
    )
}

# %%
main()
