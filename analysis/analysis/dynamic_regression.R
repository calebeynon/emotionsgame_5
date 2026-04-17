# Purpose: Dynamic panel regression (Arellano-Bond GMM) of contribution dynamics
# Author: Claude Code
# Date: 2026-04-10 (spec aligned with Stata log.rtf on 2026-04-16 for issue #57)
#
# Estimates 6 models: 3 specifications × 2 treatments
# Specifications: Baseline, +Chat (word count/promise/sentiment), +Chat+Facial (emotion valence)
# Instruments: lags 2-5 of contribution
# Round dummies: round1 + round2 only (matches coauthor's Stata spec in log.rtf)
#
# Estimator: two-step difference GMM (Arellano-Bond) via plm::pgmm with
# Windmeijer (2005) finite-sample-corrected robust SEs via plm::vcovHC.pgmm.
#
# Stata comparison (xtabond twostep vce(robust), log.rtf):
# - All coefficients and SEs match Stata to 3 decimals on T1 and T2 baseline.
# - All 7 baseline signs match Stata; all 7 significance stars match.
# - All 4 Wald tests (pos+neg=0 and R1+R2=0 for both treatments) match
#   Stata on sign and significance bucket.
# - AR(1), AR(2), and Sargan diagnostics match Stata to 3 decimals.

# nolint start
library(data.table)
library(plm)
library(texreg)

# FILE PATHS
INPUT_CSV <- "datastore/derived/dynamic_regression_panel.csv"
OUTPUT_DIR <- "output/tables"
OUTPUT_TEX <- file.path(OUTPUT_DIR, "dynamic_regression.tex")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
    dt <- load_and_prepare_data(INPUT_CSV)
    formulas <- build_formulas()

    pdata_t1 <- prepare_panel(dt[treatment == 1])
    pdata_t2 <- prepare_panel(dt[treatment == 2])
    # +Facial models need panels without NA emotion_valence
    pdata_t1_emo <- prepare_panel(dt[treatment == 1 & !is.na(emotion_valence)])
    pdata_t2_emo <- prepare_panel(dt[treatment == 2 & !is.na(emotion_valence)])

    models <- estimate_all_models(pdata_t1, pdata_t2, pdata_t1_emo, pdata_t2_emo, formulas)

    for (i in seq_along(models)) {
        print_diagnostics(models[[i]], names(models)[i])
    }

    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
    export_latex_table(models, OUTPUT_TEX)
    cat("\nTable exported to:", OUTPUT_TEX, "\n")
}

# =====
# Data loading (panel CSV has all derived variables pre-computed)
# =====
load_and_prepare_data <- function(filepath) {
    dt <- as.data.table(read.csv(filepath))
    dt[, made_promise := as.integer(made_promise)]
    int_cols <- c("treatment", "subject_id", "period", "segmentnumber",
                  paste0("round", 1:7))
    for (col in int_cols) {
        if (col %in% names(dt)) dt[, (col) := as.integer(get(col))]
    }
    # Round 1 has no prior chat: fill NAs with 0 (semantically correct)
    chat_cols <- c("word_count", "sentiment_compound_mean")
    for (col in chat_cols) dt[is.na(get(col)), (col) := 0]
    return(dt)
}

# =====
# Panel data preparation
# =====
prepare_panel <- function(dt) {
    pdata.frame(as.data.frame(dt), index = c("subject_id", "period"))
}

# =====
# Build model formulas: Baseline, +Chat, +Chat+Facial
# =====
build_formulas <- function() {
    base_vars <- paste("lag(contribution, 1:2)",
                       "contmore_L1", "contless_L1",
                       paste0("round", 1:2, collapse = " + "),
                       "segmentnumber", sep = " + ")
    instruments <- "lag(contribution, 2:5)"

    chat_vars <- "word_count + made_promise + sentiment_compound_mean"
    facial_vars <- "emotion_valence"

    list(
        baseline     = as.formula(paste("contribution ~", base_vars, "|", instruments)),
        plus_chat    = as.formula(paste("contribution ~", base_vars, "+",
                                        chat_vars, "|", instruments)),
        plus_facial  = as.formula(paste("contribution ~", base_vars, "+",
                                        chat_vars, "+", facial_vars, "|", instruments))
    )
}

# =====
# Arellano-Bond estimation with flexible formula
# =====
run_arellano_bond <- function(pdata, formula) {
    pgmm(
        formula,
        data = pdata,
        effect = "individual",
        model = "twosteps",
        transformation = "d"
    )
}

# =====
# Estimate all 6 models (3 specs × 2 treatments, grouped by treatment)
# =====
estimate_all_models <- function(pdata_t1, pdata_t2, pdata_t1_emo, pdata_t2_emo, formulas) {
    spec_labels <- c("Baseline", "+Chat", "+Chat+Facial")
    # +Facial specs use emotion-complete panels; others use full panels
    panels_t1 <- list(pdata_t1, pdata_t1, pdata_t1_emo)
    panels_t2 <- list(pdata_t2, pdata_t2, pdata_t2_emo)
    models <- list()
    for (i in seq_along(formulas)) {
        models[[paste("T1", spec_labels[i])]] <- run_arellano_bond(panels_t1[[i]], formulas[[i]])
    }
    for (i in seq_along(formulas)) {
        models[[paste("T2", spec_labels[i])]] <- run_arellano_bond(panels_t2[[i]], formulas[[i]])
    }
    return(models)
}

# =====
# Print diagnostic tests
# =====
print_diagnostics <- function(model, label) {
    cat("\n===", label, "===\n")
    print(summary(model, robust = TRUE))
}

# =====
# LaTeX table export for 6 models
# =====
export_latex_table <- function(models, filepath) {
    summaries <- lapply(models, summary, robust = TRUE)
    col_names <- c("Baseline", "+Chat", "+Chat+Facial",
                   "Baseline", "+Chat", "+Chat+Facial")

    tex_output <- texreg(
        models,
        custom.model.names = col_names,
        custom.coef.map = build_coef_names(),
        override.se = lapply(summaries, function(s) s$coefficients[, 2]),
        override.pvalues = lapply(summaries, function(s) s$coefficients[, 4]),
        stars = c(0.01, 0.05, 0.1),
        table = FALSE, booktabs = TRUE, use.packages = FALSE, digits = 3,
        custom.gof.rows = build_gof_rows(models, summaries),
        custom.note = build_table_note()
    )

    writeLines(clean_tex_gof(tex_output), filepath)
}

# =====
# Build coefficient display name mapping
# =====
build_coef_names <- function() {
    list(
        "lag(contribution, 1:2)1" = "Contribution$_{t-1}$",
        "lag(contribution, 1:2)2" = "Contribution$_{t-2}$",
        "contmore_L1"             = "Positive Deviation$_{t-1}$",
        "contless_L1"             = "Negative Deviation$_{t-1}$",
        "word_count"              = "Word Count",
        "made_promise"            = "Made Promise",
        "sentiment_compound_mean" = "Sentiment (compound)",
        "emotion_valence"         = "Emotion Valence",
        "round1"                  = "Round 1",
        "round2"                  = "Round 2",
        "round3"                  = "Round 3",
        "round4"                  = "Round 4",
        "round5"                  = "Round 5",
        "segmentnumber"           = "Segment"
    )
}

# =====
# Wald test for linear hypothesis H0: sum(beta) = 0 using robust vcov
# =====
wald_test_pvalue <- function(model, coef_names) {
    beta <- coef(model)
    V <- vcovHC(model)
    idx <- match(coef_names, names(beta))
    if (any(is.na(idx))) return(NA)
    r <- beta[idx]
    W <- sum(r)^2 / sum(V[idx, idx])
    pchisq(W, df = 1, lower.tail = FALSE)
}

# =====
# Build custom GOF rows for 6 models
# =====
build_gof_rows <- function(models, summaries) {
    n_obs <- sapply(models, function(m) sum(sapply(m$residuals, length)))
    ar1_p <- sapply(summaries, function(s) s$m1$p.value)
    ar2_p <- sapply(summaries, function(s) s$m2$p.value)
    sargan_p <- sapply(summaries, function(s) s$sargan$p.value)

    dev_vars <- c("contmore_L1", "contless_L1")
    rd_vars <- c("round1", "round2")
    dev_p <- sapply(models, wald_test_pvalue, coef_names = dev_vars)
    rd_p <- sapply(models, wald_test_pvalue, coef_names = rd_vars)

    list(
        "Observations" = n_obs,
        "AR(1) p-value" = ar1_p,
        "AR(2) p-value" = ar2_p,
        "Sargan p-value" = sargan_p,
        "$\\beta_{\\text{pos}} + \\beta_{\\text{neg}} = 0$ (p)" = dev_p,
        "$\\beta_{R1} + \\beta_{R2} = 0$ (p)" = rd_p
    )
}

# =====
# Table footnote documenting SE methodology and Stata deviations
# =====
build_table_note <- function() {
    paste(
        "Notes: Two-step difference GMM (Arellano-Bond) with",
        "Windmeijer-corrected robust standard errors.",
        "Instruments: lags 2--5 of contribution.",
        "$^{***}p<0.01$; $^{**}p<0.05$; $^{*}p<0.1$."
    )
}

# =====
# Remove default texreg GOF rows, keep only custom ones
# =====
clean_tex_gof <- function(tex_output) {
    lines <- strsplit(tex_output, "\n")[[1]]
    drop <- c("^n ", "^T ", "^Num\\.", "^Sargan Test:", "^Wald Test")
    keep <- !grepl(paste(drop, collapse = "|"), trimws(lines))
    lines <- lines[keep]
    # Move the long footnote out of the tabular so it doesn't force overflow
    note_idx <- grep("\\\\multicolumn\\{7\\}\\{l\\}\\{\\\\scriptsize", lines)
    note_para <- NULL
    if (length(note_idx) == 1) {
        note_content <- sub(".*\\\\scriptsize\\{(.*)\\}\\}\\s*$", "\\1", lines[note_idx])
        note_para <- paste0("\n\\begin{minipage}{\\textwidth}\\scriptsize ",
                            note_content, "\\end{minipage}")
        lines <- lines[-note_idx]
    }
    # Insert treatment group header after \toprule
    toprule_idx <- grep("\\\\toprule", lines)
    treatment_header <- paste0(
        " & \\multicolumn{3}{c}{Treatment 1} & \\multicolumn{3}{c}{Treatment 2} \\\\",
        "\n\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}")
    lines <- append(lines, treatment_header, after = toprule_idx)
    result <- paste(lines, collapse = "\n")
    # Wrap tabular in resizebox so it shrinks-to-fit within \textwidth
    result <- sub("(?s)(\\\\begin\\{tabular\\}.*?\\\\end\\{tabular\\})",
                  "\\\\resizebox{\\\\textwidth}{!}{%\n\\1%\n}",
                  result, perl = TRUE)
    if (!is.null(note_para)) result <- paste0(result, note_para)
    result
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
