# Purpose: Bar plots of liar and sucker count distributions across participants
# Author: Claude Code
# Date: 2026-04-10

library(data.table)
library(ggplot2)

# FILE PATHS
BEHAVIOR_CSV <- "datastore/derived/behavior_classifications.csv"
PLOT_DIR <- file.path("output", "plots")

# SHARED THEME
PLOT_THEME <- theme_minimal(base_size = 11) +
    theme(panel.grid.minor = element_blank())

# =====
# Main function
# =====
main <- function() {
    dt <- load_data()
    liar_dist <- count_liar_rounds(dt)
    sucker_dist <- count_sucker_segments(dt)
    dir.create(PLOT_DIR, recursive = TRUE, showWarnings = FALSE)
    save_barplot(liar_dist, "Rounds Lied", "liar_count_distribution.png")
    save_barplot(sucker_dist, "Segments Suckered", "sucker_count_distribution.png")
}

# =====
# Data loading
# =====
load_data <- function() {
    if (!file.exists(BEHAVIOR_CSV)) {
        stop("Missing: ", BEHAVIOR_CSV, ". Run classify_behavior.py first.")
    }
    dt <- fread(BEHAVIOR_CSV)
    dt[is.na(lied_this_round_20), lied_this_round_20 := FALSE]
    dt[is.na(is_sucker_20), is_sucker_20 := FALSE]
    return(dt)
}

# =====
# Counting
# =====
count_liar_rounds <- function(dt) {
    dt[, .(count = sum(lied_this_round_20 == TRUE)),
       by = .(session_code, label)][, .N, by = count]
}

count_sucker_segments <- function(dt) {
    per_seg <- dt[, .(suckered = any(is_sucker_20 == TRUE)),
                  by = .(session_code, label, segment)]
    per_seg[, .(count = sum(suckered)),
            by = .(session_code, label)][, .N, by = count]
}

# =====
# Plotting
# =====
save_barplot <- function(dist_dt, xlabel, filename) {
    p <- ggplot(dist_dt, aes(x = factor(count), y = N)) +
        geom_col(fill = "grey30", width = 0.7) +
        geom_text(aes(label = N), vjust = -0.5, size = 3.5) +
        labs(x = xlabel, y = "Number of Participants") +
        scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
        PLOT_THEME
    outpath <- file.path(PLOT_DIR, filename)
    ggsave(outpath, plot = p, width = 6.5, height = 4,
           units = "in", dpi = 300)
    message("Saved: ", outpath)
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
