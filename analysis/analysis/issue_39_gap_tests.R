# Purpose: Statistical tests for whether the emotion-sentiment gap differs
#          across player types (cooperative state, liar status, sucker status)
# Author: Claude Code
# Date: 2026-03-14

# nolint start
source("analysis/issue_39_common.R")
# nolint end

# OUTPUT FILE
OUTPUT_TEX <- file.path(TABLE_DIR, "emotion_sentiment_gap_tests.tex")

# =====
# Main function
# =====
main <- function() {
    dt <- prepare_data()
    results <- run_all_tests(dt)
    dir.create(TABLE_DIR, recursive = TRUE, showWarnings = FALSE)
    export_latex_table(results, OUTPUT_TEX)
    print_results(results)
    cat("Table saved to:", OUTPUT_TEX, "\n")
}

# =====
# Data preparation
# =====
prepare_data <- function() {
    dt <- load_contribute_data()
    dt <- merge_behavior_classifications(dt)
    dt <- compute_zscores(dt)
    dt <- dt[!is.na(valence_z) & !is.na(compound_z)]
    dt[, state_label := ifelse(player_state == "cooperative", "Cooperative", "Noncooperative")]
    dt[, liar_label := ifelse(is_liar_20 == TRUE, "Liar", "Honest")]
    dt[, sucker_label := ifelse(is_sucker_20 == TRUE, "Sucker", "Non-sucker")]
    dt[, liar_state := paste(liar_label, state_label, sep = " / ")]
    return(dt)
}

# =====
# T-test wrapper
# =====
gap_ttest <- function(dt, group_col, g1, g2, comparison) {
    d1 <- dt[get(group_col) == g1, zscore_gap]
    d2 <- dt[get(group_col) == g2, zscore_gap]
    if (length(d1) < 2 || length(d2) < 2) {
        stop(sprintf("Insufficient observations for '%s': n1=%d, n2=%d (need >=2 each)",
                      comparison, length(d1), length(d2)))
    }
    tt <- t.test(d1, d2)
    star <- ifelse(tt$p.value < 0.001, "***",
            ifelse(tt$p.value < 0.01, "**",
            ifelse(tt$p.value < 0.05, "*", "")))
    data.table(
        comparison = comparison, g1 = g1, g2 = g2,
        n1 = length(d1), n2 = length(d2),
        mean1 = mean(d1), mean2 = mean(d2),
        diff = mean(d1) - mean(d2),
        t_stat = as.numeric(tt$statistic),
        p_val = tt$p.value, stars = star
    )
}

# =====
# Run all comparisons
# =====
run_all_tests <- function(dt) {
    liar_dt <- dt[!is.na(is_liar_20)]
    rbindlist(list(
        gap_ttest(dt, "state_label", "Cooperative", "Noncooperative", "Cooperative State"),
        gap_ttest(dt[!is.na(is_liar_20)], "liar_label", "Honest", "Liar", "Liar Status"),
        gap_ttest(dt[!is.na(is_sucker_20)], "sucker_label", "Non-sucker", "Sucker", "Sucker Status"),
        gap_ttest(liar_dt, "liar_state", "Honest / Cooperative", "Liar / Cooperative", "Liar (Coop.)"),
        gap_ttest(liar_dt, "liar_state", "Honest / Noncooperative", "Liar / Noncooperative", "Liar (Noncoop.)")
    ))
}

# =====
# LaTeX export
# =====
export_latex_table <- function(results, filepath) {
    lines <- c(
        "\\begin{tabular}{llccccc}",
        "  \\toprule",
        "  Comparison & Group & $n$ & Mean Gap & $\\Delta$ & $t$ & $p$ \\\\",
        "  \\midrule"
    )
    for (i in seq_len(nrow(results))) {
        r <- results[i]
        lines <- c(lines, build_row_pair(r, i < nrow(results)))
    }
    lines <- c(lines, "  \\bottomrule", "\\end{tabular}")
    writeLines(lines, filepath)
}

build_row_pair <- function(r, add_space) {
    row1 <- sprintf("  %s & %s & %d & %.3f & & & \\\\", r$comparison, r$g1, r$n1, r$mean1)
    row2 <- sprintf("   & %s & %d & %.3f & %.3f & %.2f & %.4f%s \\\\",
                    r$g2, r$n2, r$mean2, r$diff, r$t_stat, r$p_val, r$stars)
    if (add_space) c(row1, row2, "  \\addlinespace") else c(row1, row2)
}

# =====
# Console output
# =====
print_results <- function(results) {
    cat("\n=== Emotion-Sentiment Gap Tests (Welch t-test) ===\n")
    for (i in seq_len(nrow(results))) {
        r <- results[i]
        cat(sprintf("  %s: diff=%.3f, t=%.2f, p=%.4f %s\n",
            r$comparison, r$diff, r$t_stat, r$p_val, r$stars))
    }
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
