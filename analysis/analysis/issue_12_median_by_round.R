# Purpose: Create median contribution vs within-segment round plot by treatment
# Author: Claude
# Date: 2026-01-25

library(data.table)
library(ggplot2)

# =====
# File paths and constants
# =====
INPUT_FILE <- "../datastore/derived/contributions.csv"
OUTPUT_DIR <- "../output/plots"
OUTPUT_FILE <- file.path(OUTPUT_DIR, "median_contribution_by_round.png")

# Color scheme for treatments
TREATMENT_COLORS <- c("1" = "#9E1B32", "2" = "#828A8F")

# =====
# Main function (shows high-level flow)
# =====
main <- function() {
    dt <- load_contributions(INPUT_FILE)
    dt_agg <- aggregate_median_by_round(dt)
    p <- create_median_plot(dt_agg)
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
aggregate_median_by_round <- function(dt) {
    # Calculate median contribution grouped by treatment and round
    # Pooled across all sessions and segments
    dt_agg <- dt[, .(median_contribution = median(contribution)),
                  by = .(treatment, round)]
    dt_agg[order(treatment, round)]
}

# =====
# Plotting
# =====
create_median_plot <- function(dt_agg) {
    ggplot(dt_agg, aes(x = round, y = median_contribution,
                       color = factor(treatment), group = treatment)) +
        geom_line(linewidth = 0.8) +
        geom_point(size = 2.5) +
        scale_x_continuous(breaks = 1:7) +
        scale_y_continuous(limits = c(0, 25), breaks = seq(0, 25, 5)) +
        scale_color_manual(values = TREATMENT_COLORS,
                          labels = c("1" = "Treatment 1", "2" = "Treatment 2")) +
        labs(x = "Round (within segment)", y = "Median Contribution", color = NULL) +
        theme_minimal() +
        theme(panel.grid.minor = element_blank(), legend.position = "bottom")
}

# =====
# Output
# =====
save_plot <- function(p, output_file) {
    dir.create(dirname(output_file), showWarnings = FALSE, recursive = TRUE)
    ggsave(output_file, p, width = 6.5, height = 4, units = "in", dpi = 300)
}

# %%
if (sys.nframe() == 0) {
    main()
}
