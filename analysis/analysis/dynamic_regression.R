# Purpose: Dynamic panel regression (Arellano-Bond GMM) of contribution dynamics
# Author: Claude Code
# Date: 2026-04-19 (issue #68: aligned with Stata issue_68_do1.do)
#
# Produces two LaTeX tables:
#   dynamic_regression_baseline.tex  - 4 cols: IF/AF x {mean, min/med/max}
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

# Upper bound on no-message NaN fills per chat column; exceeding it signals drift.
# Baseline is 800 round-1 rows (chat hasn't happened yet); 50-row headroom lets the
# guard fire on real regressions without tripping on the exact-baseline case.
MAX_NO_MESSAGE_ROUNDS <- 850

main <- function() {
    dt <- load_and_prepare_data(INPUT_CSV)
    panels <- build_all_panels(dt)
    formulas <- build_formulas()
    baseline_models <- fit_baseline_models(panels, formulas)
    extended_models <- fit_extended_models(panels, formulas)
    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
    baseline_col_names <- c("IF", "AF", "IF", "AF")
    export_table(baseline_models, BASELINE_TEX,
                 baseline_col_names, baseline_group_header())
    export_table(extended_models, EXTENDED_TEX,
                 rep(SPEC_LABELS, times = 4), extended_group_header())
    cat("\nTables written to:", BASELINE_TEX, "and", EXTENDED_TEX, "\n")
    compare_to_reference(baseline_models)
}

load_and_prepare_data <- function(filepath) {
    if (!file.exists(filepath)) {
        stop(sprintf(
            "Missing panel CSV: %s. Run build_dynamic_regression_panel.py first.",
            filepath
        ))
    }
    dt <- as.data.table(read.csv(filepath))
    dt[, made_promise := as.integer(made_promise)]
    int_cols <- c("treatment", "subject_id", "period", paste0("round", 1:7))
    for (col in int_cols) {
        if (col %in% names(dt)) dt[, (col) := as.integer(get(col))]
    }
    report_chat_nan_counts(dt)
    for (col in c("word_count", "sentiment_compound_mean")) {
        dt[is.na(get(col)), (col) := 0]
    }
    return(dt)
}

report_chat_nan_counts <- function(dt) {
    for (col in c("word_count", "sentiment_compound_mean")) {
        n_nan <- sum(is.na(dt[[col]]))
        message(sprintf("  %s NaN count: %d (bound=%d)",
                        col, n_nan, MAX_NO_MESSAGE_ROUNDS))
        if (n_nan > MAX_NO_MESSAGE_ROUNDS) {
            stop(sprintf(
                "%s has %d NaN, exceeding no-message bound %d.",
                col, n_nan, MAX_NO_MESSAGE_ROUNDS
            ))
        }
    }
}

prepare_panel <- function(dt) {
    pdata.frame(as.data.frame(dt), index = c("subject_id", "period"))
}

# if_emo / af_emo drop rows where emotion_valence is NA (AFFDEX unavailable);
# this reduces N from 1520 to ~1064 (IF) / ~1273 (AF) — used for +Chat+Facial specs only.
# Data coding: treatment == 1 is IF (Individual Feedback); treatment == 2 is AF (Aggregate Feedback).
build_all_panels <- function(dt) {
    list(
        if_    = prepare_panel(dt[treatment == 1]),
        af     = prepare_panel(dt[treatment == 2]),
        if_emo = prepare_panel(dt[treatment == 1 & !is.na(emotion_valence)]),
        af_emo = prepare_panel(dt[treatment == 2 & !is.na(emotion_valence)])
    )
}

# Formula RHS builders. round1 is the only time dummy
# (per Stata Table DP1 xtabond spec in issue_68_do1.do; round2 and segmentnumber were dropped).
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
    # lag(contribution, 2:5) as instruments matches Stata's maxldep(4) maxlags(4)
    # (see analysis/issues/issue_68_do1.do lines 112-114).
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
        "IF (mean)"        = run_arellano_bond(panels$if_, formulas$mean$base),
        "AF (mean)"        = run_arellano_bond(panels$af,  formulas$mean$base),
        "IF (min/med/max)" = run_arellano_bond(panels$if_, formulas$order$base),
        "AF (min/med/max)" = run_arellano_bond(panels$af,  formulas$order$base)
    )
}

fit_extended_models <- function(panels, formulas) {
    models <- list()
    treat_tags <- c(if_ = "IF", af = "AF")
    for (fam in c("mean", "order")) {
        fam_tag <- FAMILY_LABELS[[fam]]
        for (treat in c("if_", "af")) {
            t_tag <- treat_tags[[treat]]
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
        "contmoremax_L1"          = "Above max peer$_{t-1}$",
        "contlessmax_L1"          = "Below max peer$_{t-1}$",
        "contmoremed_L1"          = "Above median peer$_{t-1}$",
        "contlessmed_L1"          = "Below median peer$_{t-1}$",
        "contmoremin_L1"          = "Above min peer$_{t-1}$",
        "contlessmin_L1"          = "Below min peer$_{t-1}$",
        "contmore_L1"             = "Above peer mean$_{t-1}$",
        "contless_L1"             = "Below peer mean$_{t-1}$",
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
    present <- pair %in% names(coef(model))
    # Warn only on partial presence (one coef but not both); both-absent means the
    # pair belongs to a different family (e.g., min/med/max pair in a mean model).
    if (any(present) && !all(present)) {
        warning(sprintf("Wald pair %s partially present in model coefficients",
                        paste(pair, collapse = "+")))
    }
    if (!all(present)) return(NA_real_)
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
        "Peer mean pair sum = 0 (p)"   = pair_wald(c("contmore_L1", "contless_L1")),
        "Max peer pair sum = 0 (p)"    = pair_wald(c("contmoremax_L1", "contlessmax_L1")),
        "Median peer pair sum = 0 (p)" = pair_wald(c("contmoremed_L1", "contlessmed_L1")),
        "Min peer pair sum = 0 (p)"    = pair_wald(c("contmoremin_L1", "contlessmin_L1"))
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
    # Narrow tables (<8 cols) use tabular* + \extracolsep{\fill} to fill
    # \textwidth with evenly distributed column gaps. Wide tables (>=8 cols)
    # would overflow \textwidth naturally, so wrap in \resizebox instead.
    strategy <- if (length(models) >= 8) "resizebox" else "extracolsep"
    writeLines(finalize_tex(tex, length(models), group_header, strategy), filepath)
}

build_table_note <- function() {
    paste("Notes: Two-step difference GMM (Arellano-Bond) with",
          "Windmeijer-corrected robust standard errors.",
          "Instruments: lags 2--5 of contribution.",
          "$^{***}p<0.01$; $^{**}p<0.05$; $^{*}p<0.1$.")
}

extended_group_header <- function() {
    paste0(" & \\multicolumn{3}{c}{IF (mean)} & \\multicolumn{3}{c}{AF (mean)}",
           " & \\multicolumn{3}{c}{IF (min/med/max)} & \\multicolumn{3}{c}{AF (min/med/max)} \\\\\n",
           "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-13}")
}

# Groups baseline on deviation spec so col labels shrink to IF/AF; otherwise
# the wide "(min/med/max)" label stretches cols 3-4.
baseline_group_header <- function() {
    paste0(" & \\multicolumn{2}{c}{Mean deviation} & \\multicolumn{2}{c}{Min/Med/Max deviation} \\\\\n",
           "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
}

finalize_tex <- function(tex_output, ncol, group_header, strategy) {
    lines <- strsplit(tex_output, "\n")[[1]]
    drop <- c("^n ", "^T ", "^Num\\.", "^Sargan Test:", "^Wald Test",
              "\\\\scriptsize")
    lines <- lines[!grepl(paste(drop, collapse = "|"), trimws(lines))]
    body <- paste(transform_to_etable_style(lines, ncol, group_header), collapse = "\n")
    body <- wrap_body(body, strategy)
    sprintf("\\begin{minipage}{\\textwidth}\n\\scriptsize\n%s\n\\end{minipage}", body)
}

# Reshapes texreg booktabs into the fixest-etable style used by the other
# paper regression tables. Long methodology belongs in Paper.tex's \caption.
transform_to_etable_style <- function(lines, ncol, group_header) {
    top_i <- grep("\\\\toprule", lines)
    mid_idx <- grep("^\\s*\\\\midrule\\s*$", lines)
    bot_i <- grep("\\\\bottomrule", lines)
    if (!(length(top_i) == 1 && length(mid_idx) >= 2 && length(bot_i) == 1))
        stop(sprintf("transform_to_etable_style: expected 1/>=2/1 rules, got %d/%d/%d", length(top_i), length(mid_idx), length(bot_i)))
    lines[top_i] <- build_header_block(ncol, group_header)
    lines[mid_idx[1]] <- "\\midrule\n\\emph{Variables}\\\\"
    lines[mid_idx[length(mid_idx)]] <- "\\midrule\n\\emph{Fit statistics}\\\\"
    lines[bot_i] <- build_footer_block(ncol)
    lines
}

build_header_block <- function(ncol, group_header) {
    model_cols <- paste(sprintf("(%d)", seq_len(ncol)), collapse = " & ")
    dep_row <- sprintf("Dependent Variable: & \\multicolumn{%d}{c}{Contribution}\\\\",
                       ncol)
    model_row <- sprintf("Model: & %s\\\\", model_cols)
    parts <- c("\\tabularnewline \\midrule \\midrule", dep_row, model_row, "\\midrule")
    if (!is.null(group_header)) parts <- c(parts, group_header)
    paste(parts, collapse = "\n")
}

build_footer_block <- function(ncol) {
    ncells <- ncol + 1
    se_note <- "Windmeijer-corrected robust SEs in parentheses"
    signif <- "Signif. Codes: $^{***}p<0.01$; $^{**}p<0.05$; $^{*}p<0.1$"
    se_row <- sprintf("\\multicolumn{%d}{l}{\\emph{%s}}\\\\", ncells, se_note)
    signif_row <- sprintf("\\multicolumn{%d}{l}{\\emph{%s}}\\\\", ncells, signif)
    paste(c("\\midrule \\midrule", se_row, signif_row), collapse = "\n")
}

# "extracolsep" -> tabular* fills \textwidth with even gaps (for narrow tables).
# "resizebox"   -> scale tabular down to \textwidth (for tables naturally wider).
wrap_body <- function(body, strategy) {
    if (strategy == "extracolsep") {
        out <- sub("\\\\begin\\{tabular\\}\\{([^}]+)\\}",
                   "\\\\begin{tabular*}{\\\\textwidth}{@{\\\\extracolsep{\\\\fill}} \\1}", body)
        out <- sub("\\\\end\\{tabular\\}", "\\\\end{tabular*}", out)
    } else if (strategy == "resizebox") {
        out <- sub("(?s)(\\\\begin\\{tabular\\}.*?\\\\end\\{tabular\\})",
                   "\\\\resizebox{\\\\textwidth}{!}{%\n\\1%\n}", body, perl = TRUE)
    } else stop(sprintf("wrap_body: unknown strategy '%s'; expected 'extracolsep' or 'resizebox'", strategy))
    stopifnot("wrap_body: tabular regex failed" = !identical(out, body))
    out
}

# Stata Table DP1 reference values from analysis/issues/issue_68_table_dp1_reference.txt.
# Tolerance below (0.01) matches Stata's 3-decimal log output precision.
REFERENCE_ROWS <- list(
    c("IF (min/med/max)", "contmoremax_L1", 0.064),
    c("IF (min/med/max)", "contlessmax_L1", 0.071),
    c("IF (min/med/max)", "contmoremed_L1", -0.179),
    c("IF (min/med/max)", "contlessmed_L1", 0.201),
    c("IF (min/med/max)", "contmoremin_L1", -0.160),
    c("IF (min/med/max)", "contlessmin_L1", -0.016),
    c("IF (mean)", "contmore_L1", -0.406),
    c("IF (mean)", "contless_L1", 0.268),
    c("AF (mean)", "contmore_L1", -0.263),
    c("AF (mean)", "contless_L1", 0.553),
    c("IF (mean)", "round1", -12.715),
    c("AF (mean)", "round1", -5.591)
)

compare_to_reference <- function(baseline_models) {
    cat("\n=== Baseline vs Stata Table DP1 (tol=0.01) ===\n")
    any_miss <- FALSE
    for (row in REFERENCE_ROWS) {
        model_label <- row[[1]]; term <- row[[2]]; ref <- as.numeric(row[[3]])
        coef_val <- coef(baseline_models[[model_label]])[term]
        diff <- abs(coef_val - ref)
        status <- if (diff <= 0.01) "OK" else "MISS"
        if (status == "MISS") any_miss <- TRUE
        cat(sprintf("  [%s] %-20s %-18s coef=%7.3f ref=%7.3f diff=%6.3f\n",
                    status, model_label, term, coef_val, ref, diff))
    }
    if (any_miss) {
        stop("Reference comparison failed: see MISS rows above")
    }
}

# %%
# TESTING is set via <<- TRUE by dynamic_regression_validate.R before source()-ing
# this file; that suppresses main() so helper functions can be loaded for validation.
if (interactive() || !exists("TESTING")) {
    main()
}
