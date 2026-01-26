# Purpose: Create median contribution vs segment plot by treatment
# Author: Claude
# Date: 2026-01-25

library(data.table)
library(ggplot2)

# =====
# File paths and constants
# =====
INPUT_CSV <- "../datastore/derived/contributions.csv"
OUTPUT_DIR <- "../output/plots"
OUTPUT_FILE <- file.path(OUTPUT_DIR, "median_contribution_by_segment.png")

# Color scheme for treatments
TREATMENT_COLORS <- c("1" = "#9E1B32", "2" = "#828A8F")

# =====
# Main function (shows high-level flow)
# =====
main <- function() {
    dt <- load_contributions(INPUT_CSV)
    dt_agg <- aggregate_by_segment(dt)
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
aggregate_by_segment <- function(dt) {
    # Calculate median contribution pooled across sessions and rounds
    dt_agg <- dt[, .(median_contribution = median(contribution)),
                 by = .(treatment, segment)]

    # Extract segment number (1-5) for proper ordering
    dt_agg[, segment_num := as.numeric(gsub("\\D", "", segment))]
    dt_agg <- dt_agg[order(treatment, segment_num)]

    dt_agg
}

# =====
# Plotting
# =====
create_plot <- function(dt_agg) {
    ggplot(dt_agg, aes(x = segment_num, y = median_contribution,
                       group = factor(treatment), color = factor(treatment))) +
        geom_line(linewidth = 0.8) +
        geom_point(size = 2.5) +
        scale_x_continuous(breaks = 1:5, labels = 1:5) +
        scale_y_continuous(limits = c(0, 25), breaks = seq(0, 25, 5)) +
        scale_color_manual(values = TREATMENT_COLORS,
                          labels = c("1" = "Treatment 1", "2" = "Treatment 2")) +
        labs(x = "Supergame", y = "Median Contribution", color = NULL) +
        theme_minimal() +
        theme(panel.grid.minor = element_blank(), legend.position = "bottom")
}

# =====
# Output
# =====
save_plot <- function(p, output_path) {
    # Ensure output directory exists
    dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)

    ggsave(output_path, p, width = 6.5, height = 4, units = "in", dpi = 300)
    message("Saved plot to: ", output_path)
}

# %%
if (sys.nframe() == 0) {
    main()
}
