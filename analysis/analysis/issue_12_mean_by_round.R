# =============================================================================
# Purpose: Create mean contribution vs within-segment round plot by treatment
# Author: Claude
# Date: 2026-01-25
# =============================================================================

library(data.table)
library(ggplot2)

# =====
# File paths and constants
# =====
INPUT_FILE <- "../datastore/derived/contributions.csv"
OUTPUT_DIR <- "../output/plots"
OUTPUT_FILE <- file.path(OUTPUT_DIR, "mean_contribution_by_round.png")

# Color scheme for treatments
TREATMENT_COLORS <- c("1" = "#9E1B32", "2" = "#828A8F")

# =====
# Main function
# =====
main <- function() {
    dt <- load_contributions(INPUT_FILE)
    dt_agg <- aggregate_by_treatment_round(dt)
    p <- create_plot(dt_agg)
    save_plot(p, OUTPUT_FILE)
}

# =====
# Data loading
# =====
load_contributions <- function(file_path) {
    as.data.table(read.csv(file_path))
}

# =====
# Aggregation
# =====
aggregate_by_treatment_round <- function(dt) {
    # Pool across all sessions and segments, group by treatment and round
    # Compute mean, sd, n, and 95% CI
    dt_agg <- dt[, .(
        mean_contribution = mean(contribution),
        sd_contribution = sd(contribution),
        n = .N
    ), by = .(treatment, round)]

    # Calculate standard error and 95% CI bounds
    dt_agg[, se := sd_contribution / sqrt(n)]
    dt_agg[, ci_lower := mean_contribution - 1.96 * se]
    dt_agg[, ci_upper := mean_contribution + 1.96 * se]

    dt_agg <- dt_agg[order(treatment, round)]
    return(dt_agg)
}

# =====
# Plotting
# =====
create_plot <- function(dt_agg) {
    ggplot(dt_agg, aes(x = round, y = mean_contribution,
                       color = factor(treatment),
                       fill = factor(treatment),
                       group = treatment)) +
        geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
                    alpha = 0.2, color = NA) +
        geom_line(linewidth = 0.8) +
        geom_point(size = 2) +
        scale_x_continuous(breaks = 1:7) +
        scale_y_continuous(limits = c(0, 25), breaks = seq(0, 25, 5)) +
        scale_color_manual(values = TREATMENT_COLORS,
                          labels = c("1" = "Treatment 1", "2" = "Treatment 2")) +
        scale_fill_manual(values = TREATMENT_COLORS,
                         labels = c("1" = "Treatment 1", "2" = "Treatment 2")) +
        labs(x = "Round (within segment)", y = "Mean Contribution",
             color = NULL, fill = NULL) +
        theme_minimal(base_size = 12) +
        theme(
            panel.grid.minor = element_blank(),
            legend.position = "bottom"
        ) +
        guides(fill = "none")
}

# =====
# Output
# =====
save_plot <- function(p, output_file) {
    # Ensure output directory exists
    dir.create(dirname(output_file), showWarnings = FALSE, recursive = TRUE)
    ggsave(output_file, p, width = 6.5, height = 4, units = "in", dpi = 300)
    message("Plot saved to: ", output_file)
}

# =====
# Execute
# =====
if (!interactive()) {
    main()
}
