#' Mean Contribution by Segment Plot
#'
#' Creates a line/point plot showing mean contribution across supergames (1-5)
#' by treatment, pooled across all sessions and rounds.
#'
#' Author: Claude
#' Date: 2026-01-25

library(data.table)
library(ggplot2)

# =====
# File paths and constants
# =====
INPUT_FILE <- "../datastore/derived/contributions.csv"
OUTPUT_DIR <- "../output/plots"
OUTPUT_FILE <- file.path(OUTPUT_DIR, "mean_contribution_by_segment.png")

# Color scheme for treatments
TREATMENT_COLORS <- c("1" = "#9E1B32", "2" = "#828A8F")

# =====
# Main function
# =====
main <- function() {
    dt <- load_contributions(INPUT_FILE)
    dt_agg <- aggregate_by_treatment_segment(dt)
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
# Data aggregation
# =====
aggregate_by_treatment_segment <- function(dt) {
    # Compute mean contribution pooled across sessions and rounds
    dt_agg <- dt[, .(mean_contribution = mean(contribution)),
                 by = .(treatment, segment)]

    # Extract segment number for proper ordering
    dt_agg[, segment_num := as.numeric(gsub("\\D", "", segment))]
    dt_agg <- dt_agg[order(treatment, segment_num)]

    return(dt_agg)
}

# =====
# Plot creation
# =====
create_plot <- function(dt_agg) {
    ggplot(dt_agg, aes(x = segment_num, y = mean_contribution,
                       group = factor(treatment), color = factor(treatment))) +
        geom_line(linewidth = 0.8) +
        geom_point(size = 2.5) +
        scale_x_continuous(breaks = 1:5, labels = 1:5) +
        scale_y_continuous(limits = c(0, 25), breaks = seq(0, 25, 5)) +
        scale_color_manual(values = TREATMENT_COLORS,
                          labels = c("1" = "Treatment 1", "2" = "Treatment 2")) +
        labs(x = "Supergame", y = "Mean Contribution", color = NULL) +
        theme_minimal(base_size = 11) +
        theme(panel.grid.minor = element_blank(), legend.position = "bottom")
}

# =====
# Save plot
# =====
save_plot <- function(p, output_file) {
    # Ensure output directory exists
    dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)

    ggsave(output_file, p,
           width = 6.5, height = 4, units = "in", dpi = 300)

    message("Plot saved to: ", output_file)
}

# %%
if (sys.nframe() == 0) {
    main()
}
