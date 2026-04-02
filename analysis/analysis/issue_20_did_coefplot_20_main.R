# Purpose: Event study coefficient plot for < 20 threshold, main specification only
# Author: Claude Code
# Date: 2026-04-01

# nolint start
library(data.table)
library(fixest)
library(ggplot2)

# FILE PATHS
INPUT_CSV <- "datastore/derived/issue_20_did_panel.csv"
OUTPUT_DIR <- "output/plots"
OUTPUT_PLOT <- file.path(OUTPUT_DIR, "issue_20_did_coefplot_20_main.png")

# =====
# Main function
# =====
main <- function() {
    dt <- load_and_prepare_data(INPUT_CSV)
    model <- run_did_regression(dt)
    coef_df <- extract_coefs(model)

    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
    p <- create_coefplot(coef_df)
    ggsave(OUTPUT_PLOT, p, width = 6.5, height = 4, dpi = 300)
    message("Plot saved to: ", OUTPUT_PLOT)
}

# =====
# Data loading
# =====
load_and_prepare_data <- function(filepath) {
    dt <- fread(filepath)
    bool_cols <- c("got_suckered_20", "did_sample_20")
    for (col in bool_cols) {
        dt[, (col) := as.integer(as.logical(get(col)))]
    }
    dt[is.na(tau_20), tau_20 := 999L]
    dt[, tau_20 := as.integer(tau_20)]
    return(dt)
}

# =====
# Regression
# =====
run_did_regression <- function(dt) {
    dt_sub <- dt[did_sample_20 == 1]
    feols(
        contribution ~ i(tau_20, got_suckered_20, ref = c(0, 999)) + treatment | round + segment,
        data = dt_sub, cluster = ~cluster_id
    )
}

# =====
# Coefficient extraction
# =====
extract_coefs <- function(model) {
    ct <- as.data.frame(coeftable(model))
    ct$name <- rownames(ct)
    ct <- ct[grepl("::", ct$name), ]
    ct$tau <- as.integer(gsub(".*::([-0-9]+):.*", "\\1", ct$name))

    ref_row <- data.frame(
        Estimate = 0, `Std. Error` = 0, `t value` = NA,
        `Pr(>|t|)` = NA, name = "reference", tau = 0L,
        check.names = FALSE
    )
    ct <- rbind(ct, ref_row)

    data.frame(
        tau = ct$tau,
        estimate = ct$Estimate,
        ci_lower = ct$Estimate - 1.96 * ct$`Std. Error`,
        ci_upper = ct$Estimate + 1.96 * ct$`Std. Error`
    )
}

# =====
# Plot
# =====
create_coefplot <- function(coef_df) {
    ggplot(coef_df, aes(x = tau, y = estimate)) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
        geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
        geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.2) +
        geom_point(size = 2) +
        labs(
            x = expression("Rounds Since Suckered (" * tau * ")"),
            y = "Effect on Contribution"
        ) +
        theme_minimal(base_size = 12) +
        theme(panel.grid.minor = element_blank())
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
