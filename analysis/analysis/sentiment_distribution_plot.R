# Purpose: Histogram of VADER sentiment compound scores by treatment,
#          color-coded by sentiment category (negative / neutral / positive)
# Author: Claude Code
# Date: 2026-04-01

library(data.table)
library(ggplot2)

# FILE PATHS
INPUT_CSV <- "datastore/derived/sentiment_scores.csv"
PLOT_DIR <- file.path("output", "plots")
OUTPUT_T1 <- file.path(PLOT_DIR, "sentiment_distribution_t1.png")
OUTPUT_T2 <- file.path(PLOT_DIR, "sentiment_distribution_t2.png")

# SENTIMENT CATEGORY THRESHOLDS
NEUTRAL_LO <- -0.1
NEUTRAL_HI <-  0.1

# COLORS
NEG_COLOR  <- "#D32F2F"
NEU_COLOR  <- "#9E9E9E"
POS_COLOR  <- "#388E3C"

# =====
# Main
# =====
main <- function() {
    dt <- load_data()
    save_plot(build_plot(dt[treatment == "Treatment 1"], y_max = 500), OUTPUT_T1)
    save_plot(build_plot(dt[treatment == "Treatment 2"]), OUTPUT_T2)
}

# =====
# Data loading
# =====
load_data <- function() {
    dt <- fread(INPUT_CSV)
    dt[, sentiment_category := fcase(
        sentiment_compound_mean < NEUTRAL_LO, "Negative",
        sentiment_compound_mean > NEUTRAL_HI, "Positive",
        default = "Neutral"
    )]
    dt[, sentiment_category := factor(
        sentiment_category, levels = c("Negative", "Neutral", "Positive")
    )]
    dt[, treatment := factor(treatment, labels = c("Treatment 1", "Treatment 2"))]
    dt
}

# =====
# Plot construction
# =====
build_plot <- function(dt, y_max = NULL) {
    p <- ggplot(dt, aes(x = sentiment_compound_mean, fill = sentiment_category)) +
        geom_histogram(binwidth = 0.05, boundary = 0, color = "white",
                       linewidth = 0.2) +
        scale_fill_manual(
            values = c("Negative" = NEG_COLOR,
                       "Neutral"  = NEU_COLOR,
                       "Positive" = POS_COLOR)
        ) +
        labs(x = "VADER Compound Score", y = "Count") +
        theme_minimal(base_size = 11) +
        theme(
            panel.grid.minor = element_blank(),
            legend.position = "none"
        )
    if (!is.null(y_max)) {
        p <- p + coord_cartesian(ylim = c(0, y_max))
    }
    p
}

# =====
# Save
# =====
save_plot <- function(p, filepath) {
    dir.create(PLOT_DIR, recursive = TRUE, showWarnings = FALSE)
    ggsave(filepath, plot = p,
           width = 4, height = 3.5, units = "in", dpi = 300)
    message("Saved: ", filepath)
}

# %%
if (!interactive()) main()
