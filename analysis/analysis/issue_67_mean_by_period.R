# =============================================================================
# Purpose: Create mean contribution vs cumulative period plot by treatment
# Author: Claude
# Date: 2026-04-19
# =============================================================================

library(data.table)
library(ggplot2)

# =====
# File paths and constants
# =====
INPUT_FILE <- "../datastore/derived/contributions.csv"
OUTPUT_DIR <- "../output/plots"
OUTPUT_FILE <- file.path(OUTPUT_DIR, "mean_contribution_by_period.png")

# Grayscale-safe distinction: filled circle vs filled triangle
TREATMENT_SHAPES <- c("1" = 16, "2" = 17)

# Horizontal dodge so overlapping error bars stay readable
DODGE_WIDTH <- 0.35

# Literal segment lengths; do NOT infer from data
SEGMENT_ROUNDS <- c(supergame1 = 3, supergame2 = 4, supergame3 = 3,
                    supergame4 = 7, supergame5 = 5)

# Vertical dashed lines between segments
SEGMENT_BOUNDARIES <- c(3.5, 7.5, 10.5, 17.5)

# =====
# Main function
# =====
main <- function() {
    dt <- load_contributions(INPUT_FILE)
    dt <- add_period_column(dt)
    dt_agg <- aggregate_by_treatment_period(dt)
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
# Period mapping
# =====
add_period_column <- function(dt) {
    # Cumulative offset for each segment: period = offset[segment] + round
    offsets <- c(0, head(cumsum(SEGMENT_ROUNDS), -1))
    names(offsets) <- names(SEGMENT_ROUNDS)
    dt[, period := offsets[segment] + round]
    return(dt)
}

# =====
# Aggregation
# =====
aggregate_by_treatment_period <- function(dt) {
    dt_agg <- dt[, .(
        mean_contribution = mean(contribution),
        sd_contribution = sd(contribution),
        n = .N
    ), by = .(treatment, period)]

    dt_agg[, se := sd_contribution / sqrt(n)]
    dt_agg[, ci_lower := mean_contribution - 1.96 * se]
    dt_agg[, ci_upper := mean_contribution + 1.96 * se]

    dt_agg <- dt_agg[order(treatment, period)]
    return(dt_agg)
}

# =====
# Plotting
# =====
create_plot <- function(dt_agg) {
    dodge <- position_dodge(width = DODGE_WIDTH)
    ggplot(dt_agg, aes(x = period, y = mean_contribution,
                       shape = factor(treatment), group = treatment)) +
        geom_vline(xintercept = SEGMENT_BOUNDARIES,
                   linetype = "dashed", color = "gray60") +
        geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper),
                      width = 0.35, position = dodge, color = "black") +
        geom_point(position = dodge, color = "black", size = 2) +
        plot_scales_and_theme()
}

plot_scales_and_theme <- function() {
    treatment_labels <- c("1" = "Treatment 1", "2" = "Treatment 2")
    list(
        scale_x_continuous(breaks = 1:22,
                           expand = expansion(mult = c(0.01, 0.01))),
        scale_y_continuous(breaks = seq(0, 25, 5)),
        coord_cartesian(ylim = c(0, 25)),
        scale_shape_manual(values = TREATMENT_SHAPES, labels = treatment_labels),
        labs(x = "Round", y = "Mean Contribution", shape = NULL),
        theme_econ_local(),
        theme(legend.position = "bottom")
    )
}

theme_econ_local <- function() {
    theme_minimal(base_size = 12) +
        theme(
            panel.grid.minor = element_blank(),
            panel.grid.major = element_line(color = "gray90"),
            text = element_text(family = "serif"),
            axis.text = element_text(size = 10),
            axis.title = element_text(size = 11)
        )
}

# =====
# Output
# =====
save_plot <- function(p, output_file) {
    dir.create(dirname(output_file), showWarnings = FALSE, recursive = TRUE)
    ggsave(output_file, p, width = 7, height = 4, units = "in", dpi = 300)
    message("Plot saved to: ", output_file)
}

# =====
# Execute
# =====
if (!interactive()) {
    main()
}
