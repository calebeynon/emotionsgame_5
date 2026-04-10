# Purpose: Generate dot plots of detrended valence (segment-mean and reverse)
#          by liar/sucker status during the Chat period
# Author: Claude Code
# Date: 2026-04-09

# nolint start
source("analysis/issue_52_common.R")
# nolint end

# OUTPUT DIRECTORY
DETREND_PLOT_DIR <- file.path(PLOT_DIR, "within_person_detrended_chat")

# COLORS (matching within-person deviation plots)
DETREND_COLOR <- "#2166AC"

# =====
# Main function
# =====
main <- function() {
    dt <- load_chat_emotion_data_detrended()
    results <- rbindlist(list(
        generate_method_plots(dt, "valence_segmean_detrended", "Segment-Mean"),
        generate_method_plots(dt, "valence_reverse_detrended", "Reverse")
    ))
    print_detrend_summary(results)
}

# =====
# Generate 4 plots for one detrending method
# =====
generate_method_plots <- function(dt, col, method_label) {
    specs <- build_detrend_specs(method_label)
    dt_valid <- dt[!is.na(get(col))]
    summaries <- mapply(
        function(filter_col, group_col, filename) {
            save_detrend_plot(dt_valid, col, method_label,
                              filter_col, group_col, filename)
        },
        specs$filter, specs$group, specs$file, SIMPLIFY = FALSE
    )
    rbindlist(summaries)
}

# =====
# Define the 4 plot specifications per method
# =====
build_detrend_specs <- function(method_label) {
    prefix <- gsub("-", "_", tolower(method_label))
    data.table(
        filter = c("is_liar_20", "is_sucker_20",
                    "is_liar_20", "is_sucker_20"),
        group  = c("liar_label", "sucker_label",
                    "first_time_liar_label", "first_time_sucker_label"),
        file   = c(
            sprintf("valence_%s_by_liar_status.png", prefix),
            sprintf("valence_%s_by_sucker_status.png", prefix),
            sprintf("valence_%s_by_first_time_liar.png", prefix),
            sprintf("valence_%s_by_first_time_sucker.png", prefix)
        )
    )
}

# =====
# Save one plot and return summary row
# =====
save_detrend_plot <- function(dt, col, method_label,
                               filter_col, group_col, filename) {
    dt_sub <- dt[!is.na(get(filter_col))]
    x_label <- paste0("Valence (", method_label, " Detrended)")
    p <- build_detrend_dotplot(dt_sub, col, group_col, x_label)
    save_plot(p, filename, subdir = DETREND_PLOT_DIR)
    summarize_detrend(dt_sub, col, group_col, method_label, filename)
}

# =====
# Build horizontal dot plot with 95% CIs
# =====
build_detrend_dotplot <- function(dt, col, group_col, x_label) {
    s <- dt[, .(mean = mean(get(col)),
                se = sd(get(col)) / sqrt(.N),
                n = .N), by = group_col]
    setnames(s, group_col, "group")
    s[, lower := mean - 1.96 * se]
    s[, upper := mean + 1.96 * se]
    s[, group_n := paste0(group, " (n=", n, ")")]

    ggplot(s, aes(x = mean, y = group_n)) +
        geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
        geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0.15,
                       color = DETREND_COLOR) +
        geom_point(size = 2.5, color = DETREND_COLOR) +
        labs(x = x_label, y = NULL) +
        PLOT_THEME
}

# =====
# Summary statistics by group
# =====
summarize_detrend <- function(dt, col, group_col,
                               method_label, filename) {
    s <- dt[, .(mean = mean(get(col)),
                se = sd(get(col)) / sqrt(.N),
                n = .N), by = group_col]
    setnames(s, group_col, "group")
    s[, lower := mean - 1.96 * se]
    s[, upper := mean + 1.96 * se]
    sig <- any(s$lower > 0 | s$upper < 0, na.rm = TRUE)
    data.table(method = method_label, plot = filename, significant = sig)
}

# =====
# Print summary of significant results
# =====
print_detrend_summary <- function(results_dt) {
    message(sprintf("\n=== Summary: %d detrended plots generated ===",
                    nrow(results_dt)))
    message(sprintf("Output directory: %s", DETREND_PLOT_DIR))
    sig_rows <- results_dt[significant == TRUE]
    if (nrow(sig_rows) > 0) {
        message("Plots with CI excluding zero:")
        for (i in seq_len(nrow(sig_rows))) {
            message(sprintf("  - [%s] %s", sig_rows$method[i],
                            sig_rows$plot[i]))
        }
    } else {
        message("No plots had group CIs excluding zero.")
    }
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
