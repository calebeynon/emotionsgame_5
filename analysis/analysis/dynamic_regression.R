# Purpose: Dynamic panel regression (Arellano-Bond GMM) of contribution dynamics
# Author: Claude Code
# Date: 2026-04-19 (issue #68: aligned with Stata issue_68_do1.do)
#
# Produces two LaTeX tables:
#   dynamic_regression_baseline.tex  - 4 cols: T1/T2 x {mean, min/med/max}
#   dynamic_regression_extended.tex  - 12 cols: 4 baselines x {Base, +Chat, +Chat+Facial}
#
# Instruments: lag(contribution, 2:5). Estimator: two-step difference GMM (pgmm)
# with Windmeijer-corrected robust SEs (vcovHC.pgmm).
# nolint start
library(data.table)
library(plm)
library(texreg)

# FILE PATHS
INPUT_CSV <- "datastore/derived/dynamic_regression_panel.csv"
OUTPUT_DIR <- "output/tables"
BASELINE_TEX <- file.path(OUTPUT_DIR, "dynamic_regression_baseline.tex")
EXTENDED_TEX <- file.path(OUTPUT_DIR, "dynamic_regression_extended.tex")

FAMILY_LABELS <- c(mean = "mean", order = "min/med/max")
SPEC_LABELS <- c("Base", "+Chat", "+Chat+Facial")

main <- function() {
    dt <- load_and_prepare_data(INPUT_CSV)
    panels <- build_all_panels(dt)
    formulas <- build_formulas()
    baseline_models <- fit_baseline_models(panels, formulas)
    extended_models <- fit_extended_models(panels, formulas)
    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
    export_table(baseline_models, BASELINE_TEX, names(baseline_models), NULL)
    export_table(extended_models, EXTENDED_TEX,
                 rep(SPEC_LABELS, times = 4), extended_group_header())
    cat("\nTables written to:", BASELINE_TEX, "and", EXTENDED_TEX, "\n")
    compare_to_reference(baseline_models)
}

load_and_prepare_data <- function(filepath) {
    dt <- as.data.table(read.csv(filepath))
    dt[, made_promise := as.integer(made_promise)]
    int_cols <- c("treatment", "subject_id", "period", paste0("round", 1:7))
    for (col in int_cols) {
        if (col %in% names(dt)) dt[, (col) := as.integer(get(col))]
    }
    for (col in c("word_count", "sentiment_compound_mean")) {
        dt[is.na(get(col)), (col) := 0]
    }
    return(dt)
}

prepare_panel <- function(dt) {
    pdata.frame(as.data.frame(dt), index = c("subject_id", "period"))
}

build_all_panels <- function(dt) {
    list(
        t1     = prepare_panel(dt[treatment == 1]),
        t2     = prepare_panel(dt[treatment == 2]),
        t1_emo = prepare_panel(dt[treatment == 1 & !is.na(emotion_valence)]),
        t2_emo = prepare_panel(dt[treatment == 2 & !is.na(emotion_valence)])
    )
}

# Formula builders (no round2, no segmentnumber)
mean_base_rhs <- function() {
    paste("lag(contribution, 1:2)", "contmore_L1", "contless_L1", "round1",
          sep = " + ")
}

order_base_rhs <- function() {
    paste("lag(contribution, 1:2)",
          "contmoremax_L1", "contlessmax_L1",
          "contmoremed_L1", "contlessmed_L1",
          "contmoremin_L1", "contlessmin_L1",
          "round1", sep = " + ")
}

build_formulas <- function() {
    instr <- "lag(contribution, 2:5)"
    chat <- "word_count + made_promise + sentiment_compound_mean"
    facial <- "emotion_valence"
    list(
        mean  = assemble_spec_formulas(mean_base_rhs(), chat, facial, instr),
        order = assemble_spec_formulas(order_base_rhs(), chat, facial, instr)
    )
}

assemble_spec_formulas <- function(base_rhs, chat, facial, instr) {
    f <- function(rhs) as.formula(paste("contribution ~", rhs, "|", instr))
    list(
        base   = f(base_rhs),
        chat   = f(paste(base_rhs, "+", chat)),
        facial = f(paste(base_rhs, "+", chat, "+", facial))
    )
}

run_arellano_bond <- function(pdata, formula) {
    pgmm(formula, data = pdata, effect = "individual",
         model = "twosteps", transformation = "d")
}

fit_baseline_models <- function(panels, formulas) {
    list(
        "T1 (mean)"        = run_arellano_bond(panels$t1, formulas$mean$base),
        "T2 (mean)"        = run_arellano_bond(panels$t2, formulas$mean$base),
        "T1 (min/med/max)" = run_arellano_bond(panels$t1, formulas$order$base),
        "T2 (min/med/max)" = run_arellano_bond(panels$t2, formulas$order$base)
    )
}

fit_extended_models <- function(panels, formulas) {
    models <- list()
    for (fam in c("mean", "order")) {
        fam_tag <- FAMILY_LABELS[[fam]]
        for (treat in c("t1", "t2")) {
            t_tag <- toupper(sub("t", "T", treat))
            emo_panel <- panels[[paste0(treat, "_emo")]]
            panel_list <- list(panels[[treat]], panels[[treat]], emo_panel)
            for (i in seq_along(SPEC_LABELS)) {
                spec_tag <- names(formulas[[fam]])[i]
                label <- sprintf("%s %s %s", t_tag, fam_tag, SPEC_LABELS[i])
                models[[label]] <- run_arellano_bond(
                    panel_list[[i]], formulas[[fam]][[spec_tag]]
                )
            }
        }
    }
    return(models)
}

build_coef_names <- function() {
    list(
        "lag(contribution, 1:2)1" = "Contribution$_{t-1}$",
        "lag(contribution, 1:2)2" = "Contribution$_{t-2}$",
        "contmoremax_L1"          = "contmoremax$_{t-1}$",
        "contlessmax_L1"          = "contlessmax$_{t-1}$",
        "contmoremed_L1"          = "contmoremed$_{t-1}$",
        "contlessmed_L1"          = "contlessmed$_{t-1}$",
        "contmoremin_L1"          = "contmoremin$_{t-1}$",
        "contlessmin_L1"          = "contlessmin$_{t-1}$",
        "contmore_L1"             = "Positive Deviation$_{t-1}$",
        "contless_L1"             = "Negative Deviation$_{t-1}$",
        "word_count"              = "Word Count",
        "made_promise"            = "Made Promise",
        "sentiment_compound_mean" = "Sentiment (compound)",
        "emotion_valence"         = "Emotion Valence",
        "round1"                  = "Round 1"
    )
}

wald_test_pvalue <- function(model, coef_names) {
    beta <- coef(model)
    V <- vcovHC(model)
    idx <- match(coef_names, names(beta))
    if (any(is.na(idx))) {
        missing <- coef_names[is.na(idx)]
        stop(sprintf("wald_test_pvalue: coef(s) not in model: %s",
                     paste(missing, collapse = ", ")))
    }
    r <- beta[idx]
    W <- sum(r)^2 / sum(V[idx, idx])
    pchisq(W, df = 1, lower.tail = FALSE)
}

model_wald_sum <- function(model, pair) {
    if (!all(pair %in% names(coef(model)))) return(NA_real_)
    wald_test_pvalue(model, pair)
}

build_gof_rows <- function(models, summaries) {
    n_obs <- sapply(models, function(m) sum(sapply(m$residuals, length)))
    ar1_p <- sapply(summaries, function(s) s$m1$p.value)
    ar2_p <- sapply(summaries, function(s) s$m2$p.value)
    sargan_p <- sapply(summaries, function(s) s$sargan$p.value)
    pair_wald <- function(pair) sapply(models, model_wald_sum, pair = pair)
    list(
        "Observations"       = n_obs,
        "AR(1) p-value"      = ar1_p,
        "AR(2) p-value"      = ar2_p,
        "Sargan p-value"     = sargan_p,
        "pos+neg=0 (p)"      = pair_wald(c("contmore_L1", "contless_L1")),
        "max$^+$+max$^-$=0 (p)" = pair_wald(c("contmoremax_L1", "contlessmax_L1")),
        "med$^+$+med$^-$=0 (p)" = pair_wald(c("contmoremed_L1", "contlessmed_L1")),
        "min$^+$+min$^-$=0 (p)" = pair_wald(c("contmoremin_L1", "contlessmin_L1"))
    )
}

export_table <- function(models, filepath, col_names, group_header) {
    summaries <- lapply(models, summary, robust = TRUE)
    tex <- texreg(
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
    writeLines(finalize_tex(tex, length(models), group_header), filepath)
}

build_table_note <- function() {
    paste("Notes: Two-step difference GMM (Arellano-Bond) with",
          "Windmeijer-corrected robust standard errors.",
          "Instruments: lags 2--5 of contribution.",
          "$^{***}p<0.01$; $^{**}p<0.05$; $^{*}p<0.1$.")
}

extended_group_header <- function() {
    paste0(
        " & \\multicolumn{3}{c}{T1 (mean)} & \\multicolumn{3}{c}{T2 (mean)}",
        " & \\multicolumn{3}{c}{T1 (min/med/max)}",
        " & \\multicolumn{3}{c}{T2 (min/med/max)} \\\\",
        "\n\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
        " \\cmidrule(lr){8-10} \\cmidrule(lr){11-13}")
}

finalize_tex <- function(tex_output, ncol, group_header) {
    lines <- strsplit(tex_output, "\n")[[1]]
    drop <- c("^n ", "^T ", "^Num\\.", "^Sargan Test:", "^Wald Test")
    lines <- lines[!grepl(paste(drop, collapse = "|"), trimws(lines))]
    parts <- extract_footnote(lines, ncol)
    body_lines <- parts$body
    if (!is.null(group_header)) body_lines <- insert_after_toprule(body_lines, group_header)
    paste0(wrap_tabular(paste(body_lines, collapse = "\n")), parts$note)
}

extract_footnote <- function(lines, ncol) {
    pat <- sprintf("\\\\multicolumn\\{%d\\}\\{l\\}\\{\\\\scriptsize", ncol + 1)
    note_idx <- grep(pat, lines)
    stopifnot("extract_footnote: expected one footnote row" = length(note_idx) == 1)
    note_content <- sub(".*\\\\scriptsize\\{(.*)\\}\\}\\s*$", "\\1", lines[note_idx])
    note <- paste0("\n\\begin{minipage}{\\textwidth}\\scriptsize ",
                   note_content, "\\end{minipage}")
    list(body = lines[-note_idx], note = note)
}

insert_after_toprule <- function(lines, header) {
    toprule_idx <- grep("\\\\toprule", lines)
    stopifnot("insert_after_toprule: expected one toprule" = length(toprule_idx) == 1)
    append(lines, header, after = toprule_idx)
}

wrap_tabular <- function(body) {
    wrapped <- sub("(?s)(\\\\begin\\{tabular\\}.*?\\\\end\\{tabular\\})",
                   "\\\\resizebox{\\\\textwidth}{!}{%\n\\1%\n}",
                   body, perl = TRUE)
    stopifnot("wrap_tabular: resizebox regex failed" = !identical(wrapped, body))
    wrapped
}

# Stata Table DP1 reference anchors for sanity verification.
REFERENCE_ROWS <- list(
    c("T1 (min/med/max)", "contmoremax_L1", 0.064),
    c("T1 (min/med/max)", "contlessmax_L1", 0.071),
    c("T1 (min/med/max)", "contmoremed_L1", -0.179),
    c("T1 (min/med/max)", "contlessmed_L1", 0.201),
    c("T1 (min/med/max)", "contmoremin_L1", -0.160),
    c("T1 (min/med/max)", "contlessmin_L1", -0.016),
    c("T1 (mean)", "contmore_L1", -0.406),
    c("T1 (mean)", "contless_L1", 0.268),
    c("T2 (mean)", "contmore_L1", -0.263),
    c("T2 (mean)", "contless_L1", 0.553),
    c("T1 (mean)", "round1", -12.715),
    c("T2 (mean)", "round1", -5.591)
)

compare_to_reference <- function(baseline_models) {
    cat("\n=== Baseline vs Stata Table DP1 (tol=0.01) ===\n")
    for (row in REFERENCE_ROWS) {
        model_label <- row[[1]]; term <- row[[2]]; ref <- as.numeric(row[[3]])
        coef_val <- coef(baseline_models[[model_label]])[term]
        diff <- abs(coef_val - ref)
        status <- if (diff <= 0.01) "OK" else "MISS"
        cat(sprintf("  [%s] %-20s %-18s coef=%7.3f ref=%7.3f diff=%6.3f\n",
                    status, model_label, term, coef_val, ref, diff))
    }
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
