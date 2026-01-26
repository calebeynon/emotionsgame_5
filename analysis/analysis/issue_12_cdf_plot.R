#' Purpose: Create CDF plot of contributions by treatment
#' Author: Claude
#' Date: 2026-01-25

library(data.table)
library(ggplot2)

# =====
# File paths and constants
# =====
INPUT_FILE <- "../datastore/derived/contributions.csv"
OUTPUT_DIR <- "../output/plots"
OUTPUT_FILE <- file.path(OUTPUT_DIR, "contribution_cdf_by_treatment.png")

# Color scheme for treatments
TREATMENT_COLORS <- c("Treatment 1" = "#9E1B32", "Treatment 2" = "#828A8F")

# =====
# Main function
# =====
main <- function() {
    dt <- load_contributions(INPUT_FILE)
    p <- create_cdf_plot(dt)
    save_plot(p, OUTPUT_FILE, OUTPUT_DIR)
}

# =====
# Data loading
# =====
load_contributions <- function(file_path) {
    dt <- fread(file_path)
    dt[, treatment := factor(treatment, labels = c("Treatment 1", "Treatment 2"))]
    return(dt)
}

# =====
# Plot creation
# =====
create_cdf_plot <- function(dt) {
    p <- ggplot(dt, aes(x = contribution, color = treatment)) +
        stat_ecdf(geom = "step", linewidth = 0.8) +
        scale_x_continuous(limits = c(0, 25), breaks = seq(0, 25, 5)) +
        scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
        scale_color_manual(values = TREATMENT_COLORS) +
        labs(x = "Contribution", y = "Cumulative Probability", color = NULL) +
        theme_minimal(base_size = 11) +
        theme(panel.grid.minor = element_blank(), legend.position = "bottom")
    return(p)
}

# =====
# Plot saving
# =====
save_plot <- function(p, output_file, output_dir) {
    if (!dir.exists(output_dir)) {
        dir.create(output_dir, recursive = TRUE)
    }
    ggsave(
        output_file,
        plot = p,
        width = 6.5,
        height = 4,
        units = "in",
        dpi = 300
    )
}

# %%
if (sys.nframe() == 0) {
    main()
}
