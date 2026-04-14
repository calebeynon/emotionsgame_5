# Purpose: Face-valence regressions for all 13 AFFDEX emotions. Produces
#          two summary tables (Lied, Suckered) with rows = emotions and
#          columns = face window specs (Results Page, Pre-Decision Chat).
# Author: Claude Code
# Date: 2026-04-13
library(fixest)
library(data.table)
# nolint start
source("analysis/issue_52_common.R")
# nolint end

# OUTPUT
GAP_SUMMARY_LIED_FILE <- file.path(
    TABLE_DIR, "issue_52_gap_summary_lied.tex")
GAP_SUMMARY_SUCKERED_FILE <- file.path(
    TABLE_DIR, "issue_52_gap_summary_suckered.tex")

SPEC_LABELS <- c(
    results = "Results Page Face",
    pre_chat = "Pre-Decision Chat Face"
)

# =====
# Main — estimate both specs for all 13 emotions, export 2 summary tables
# =====
main <- function() {
    emo_cols <- names(EMOTION_DISPLAY_NAMES)
    all_results <- lapply(emo_cols, estimate_all_specs_for_emotion)
    names(all_results) <- emo_cols
    build_summary_table_tex(all_results, "lied", GAP_SUMMARY_LIED_FILE)
    build_summary_table_tex(all_results, "suckered",
                            GAP_SUMMARY_SUCKERED_FILE)
    print_coefficient_matrix(all_results)
}

# =====
# Estimate both face specs for a given emotion column
# =====
estimate_all_specs_for_emotion <- function(emo_col) {
    message(sprintf("=== %s ===", emo_col))
    list(
        results = estimate_face_models_emo(
            prepare_results_face_data_emo(emo_col), emo_col),
        pre_chat = estimate_face_models_emo(
            prepare_pre_chat_face_data_emo(emo_col),
            paste0(emo_col, "_shifted"))
    )
}

# =====
# Prepare results-page face rows for one emotion
# =====
prepare_results_face_data_emo <- function(emo_col) {
    dt <- load_results_emotion_data()
    dt[, player_id := paste0(session_code, "_", label)]
    dt <- add_suckered_this_round(dt)
    dt <- dt[!is.na(get(emo_col))]
    return(dt)
}

# =====
# Prepare pre-decision chat face rows for one emotion (shifted by 1)
# =====
prepare_pre_chat_face_data_emo <- function(emo_col) {
    dt <- load_chat_emotion_data()
    dt[, player_id := paste0(session_code, "_", label)]
    dt <- add_suckered_this_round(dt)
    shifted <- paste0(emo_col, "_shifted")
    setorderv(dt, c("session_code", "label", "segment", "round"))
    dt[, (shifted) := shift(get(emo_col), n = 1, type = "lag"),
       by = .(session_code, label, segment)]
    dt <- dt[!is.na(get(shifted))]
    return(dt)
}

# =====
# Derive suckered-this-round from behavior CSV
# =====
add_suckered_this_round <- function(dt) {
    bc <- fread(BEHAVIOR_CSV)
    gh <- bc[lied_this_round_20 == TRUE,
             .(groupmate_lied = TRUE),
             by = .(session_code, segment, round, group)]
    dt <- merge(dt, gh,
                by = c("session_code", "segment", "round", "group"),
                all.x = TRUE)
    dt[is.na(groupmate_lied), groupmate_lied := FALSE]
    dt[, suckered_this_round := groupmate_lied &
           (contribution == 25) & (lied_this_round_20 == FALSE)]
    dt[, groupmate_lied := NULL]
    return(dt)
}

# =====
# Face-only lied/suckered models for one emotion column
# =====
estimate_face_models_emo <- function(dt, y_col) {
    dt <- copy(dt)
    dt[, round := factor(round)]
    fml_lied <- as.formula(paste0(
        y_col, " ~ lied_this_round_20 + i(round)",
        " | segment + player_id"))
    fml_suck <- as.formula(paste0(
        y_col, " ~ suckered_this_round + i(round)",
        " | segment + player_id"))
    list(
        lied = feols(fml_lied, cluster = ~player_id, data = dt),
        suckered = feols(fml_suck, cluster = ~player_id, data = dt)
    )
}

# =====
# Extract main effect for lied or suckered flag from a fitted feols model
# =====
extract_flag_coef <- function(model, flag) {
    ct <- coeftable(model)
    pattern <- paste0("^", flag, "TRUE$")
    row <- which(grepl(pattern, rownames(ct)))
    if (length(row) == 0) {
        stop(sprintf("extract_flag_coef: no row matching %s", pattern))
    }
    list(coef = ct[row, 1], se = ct[row, 2],
         p_value = ct[row, 4], nobs = model$nobs)
}

# =====
# Build and write a LaTeX summary table for one flag
# =====
build_summary_table_tex <- function(all_results, flag_key, outfile) {
    dir.create(dirname(outfile), recursive = TRUE, showWarnings = FALSE)
    spec_keys <- names(SPEC_LABELS)
    flag_name <- if (flag_key == "lied") "Lied" else "Suckered"
    lines <- latex_header_lines(flag_name)
    for (emo in names(all_results)) {
        lines <- c(lines, emotion_row_lines(all_results[[emo]], emo,
                                             flag_key, spec_keys))
    }
    lines <- c(lines, latex_footer_lines(all_results, flag_key, spec_keys))
    writeLines(lines, outfile)
    message(sprintf("Table saved: %s", outfile))
}

latex_header_lines <- function(flag_name) {
    n_specs <- length(SPEC_LABELS)
    col_spec <- paste0("l", paste(rep("c", n_specs), collapse = ""))
    c("\\begingroup",
      "\\centering",
      sprintf("\\begin{tabular}{%s}", col_spec),
      "   \\tabularnewline \\midrule \\midrule",
      sprintf("   Coefficient: & \\multicolumn{%d}{c}{%s}\\\\",
              n_specs, flag_name),
      sprintf("    & %s \\\\", paste(SPEC_LABELS, collapse = " & ")),
      "   \\midrule")
}

emotion_row_lines <- function(emo_results, emo_col, flag_key, spec_keys) {
    flag_token <- if (flag_key == "lied") "lied_this_round_20" else
                  "suckered_this_round"
    coefs <- lapply(spec_keys, function(sk) {
        extract_flag_coef(emo_results[[sk]][[flag_key]], flag_token)
    })
    pretty <- EMOTION_DISPLAY_NAMES[[emo_col]]
    coef_row <- sprintf("   %s & %s\\\\", pretty,
                        paste(sapply(coefs, format_coef_line),
                              collapse = " & "))
    se_row <- sprintf("    & %s\\\\",
                      paste(sapply(coefs, format_se_line), collapse = " & "))
    c(coef_row, se_row)
}

latex_footer_lines <- function(all_results, flag_key, spec_keys) {
    first_emo <- names(all_results)[1]
    nobs_vec <- sapply(spec_keys, function(sk) {
        all_results[[first_emo]][[sk]][[flag_key]]$nobs
    })
    nobs_row <- sprintf("   Observations & %s\\\\",
                        paste(format(nobs_vec, big.mark = ","),
                              collapse = " & "))
    note_span <- length(SPEC_LABELS) + 1
    c("   \\midrule",
      nobs_row,
      "   \\midrule \\midrule",
      "\\end{tabular}",
      sprintf("\\multicolumn{%d}{l}{\\emph{Clustered (player\\_id) SEs in parens}}\\\\",
              note_span),
      sprintf("\\multicolumn{%d}{l}{\\emph{Signif. Codes: ***: 0.01, **: 0.05, *: 0.1}}",
              note_span),
      "\\par\\endgroup")
}

format_coef_line <- function(coef) {
    sprintf("%.3f%s", coef$coef, significance_stars(coef$p_value))
}

format_se_line <- function(coef) {
    sprintf("(%.3f)", coef$se)
}

significance_stars <- function(p) {
    if (is.na(p)) return("")
    if (p < 0.01) return("$^{***}$")
    if (p < 0.05) return("$^{**}$")
    if (p < 0.10) return("$^{*}$")
    return("")
}

# =====
# Print a compact 13 x 4 matrix of coefficients
# =====
print_coefficient_matrix <- function(all_results) {
    message("\n=== Coefficient matrix (emotion x {spec,flag}) ===")
    spec_keys <- names(SPEC_LABELS)
    header <- c("Emotion",
                paste0(spec_keys, "_lied"), paste0(spec_keys, "_suck"))
    message(paste(header, collapse = " | "))
    for (emo in names(all_results)) {
        row_vals <- build_matrix_row(all_results[[emo]], spec_keys)
        message(paste(c(EMOTION_DISPLAY_NAMES[[emo]], row_vals),
                      collapse = " | "))
    }
}

build_matrix_row <- function(emo_results, spec_keys) {
    lied <- sapply(spec_keys, function(sk) {
        sprintf("%+.3f", extract_flag_coef(
            emo_results[[sk]]$lied, "lied_this_round_20")$coef)
    })
    suck <- sapply(spec_keys, function(sk) {
        sprintf("%+.3f", extract_flag_coef(
            emo_results[[sk]]$suckered, "suckered_this_round")$coef)
    })
    c(lied, suck)
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
