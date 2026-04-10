# Purpose: Estimate conditional probability of lying via logistic regression
#   P(liar_t | liar_{t-1}) with gender and treatment controls, segment FE
# Author: Claude Code
# Date: 2026-04-09

library(data.table)
library(fixest)

# FILE PATHS
BEHAVIOR_CSV <- "datastore/derived/behavior_classifications.csv"
RAW_DIR <- "datastore/raw"
OUTPUT_DIR <- "output/tables"
OUTPUT_TEX <- file.path(OUTPUT_DIR, "liar_conditional_probability.tex")

# Session code remap: raw 03_t2 valid rows use z8dowljr, derived uses irrzlgk2
SESSION_CODE_REMAP <- c(z8dowljr = "irrzlgk2")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
    dt <- load_behavior_data()
    gender <- load_gender_from_raw()
    dt <- merge_gender(dt, gender)
    dt <- create_lag(dt)

    cat(sprintf("Regression sample: %d obs, %d participants\n",
                nrow(dt), uniqueN(dt$label_session)))

    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

    models <- estimate_models(dt)
    export_table(models, OUTPUT_TEX)
    cat("Table exported to:", OUTPUT_TEX, "\n")
}

# =====
# Data loading
# =====
load_behavior_data <- function() {
    dt <- as.data.table(read.csv(BEHAVIOR_CSV))
    dt[, lied := as.integer(get("lied_this_round_20") == "True")]
    dt[, cluster_id := paste(session_code, segment, group, sep = "_")]
    dt[, label_session := paste(label, session_code, sep = "_")]
    return(dt)
}

load_gender_from_raw <- function() {
    files <- list.files(RAW_DIR, pattern = "_data\\.csv$", full.names = TRUE)
    if (length(files) == 0) stop("No *_data.csv files in ", RAW_DIR)
    gender_list <- lapply(files, extract_gender_from_file)
    gender_dt <- rbindlist(gender_list)
    cat(sprintf("Gender data: %d participants from %d files\n",
                nrow(gender_dt), length(files)))
    return(gender_dt)
}

extract_gender_from_file <- function(filepath) {
    raw <- fread(filepath, encoding = "UTF-8")
    dt <- raw[, .(
        session_code = get("session.code"),
        label = get("participant.label"),
        gender = get("finalresults.1.player.q1")
    )]
    # Special case: 03_t2 has two session codes, only z8dowljr is valid
    if ("z8dowljr" %in% dt$session_code) {
        dt <- dt[session_code == "z8dowljr"]
    }
    # Apply session code remap
    for (old_code in names(SESSION_CODE_REMAP)) {
        dt[session_code == old_code, session_code := SESSION_CODE_REMAP[[old_code]]]
    }
    return(dt)
}

# =====
# Gender merge
# =====
merge_gender <- function(dt, gender) {
    n_before <- nrow(dt)
    dt <- merge(dt, gender, by = c("session_code", "label"), all.x = TRUE)
    n_missing <- sum(is.na(dt$gender))
    if (n_missing > 0) {
        warning(sprintf("Gender missing for %d of %d rows", n_missing, n_before))
    }
    dt[, female := as.integer(gender == "Female")]
    return(dt)
}

# =====
# Lag creation (within participant x segment)
# =====
create_lag <- function(dt) {
    setorder(dt, label_session, segment, round)
    dt[, lied_prev_round := shift(lied, 1, type = "lag"),
       by = .(label_session, segment)]
    # Drop first round of each segment (no lag available)
    dt <- dt[!is.na(lied_prev_round)]
    cat(sprintf("After dropping first rounds: %d obs\n", nrow(dt)))
    return(dt)
}

# =====
# Regression models
# =====
estimate_models <- function(dt) {
    m1 <- run_logit(dt, "lied ~ lied_prev_round + female + treatment | segment")
    print(summary(m1))
    return(list(m1 = m1))
}

run_logit <- function(dt, formula_str) {
    feglm(
        as.formula(formula_str),
        data = dt,
        family = binomial(link = "logit"),
        cluster = ~label_session
    )
}

# =====
# LaTeX export
# =====
export_table <- function(models, filepath) {
    etable(
        models$m1,
        file = filepath,
        tex = TRUE,
        fitstat = c("n"),
        dict = c(
            lied_prev_round = "Lied Previous Round",
            female = "Female",
            treatment = "Treatment"
        ),
        title = "Conditional Probability of Lying (Logistic Regression)",
        se.below = TRUE,
        postprocess.tex = function(x) {
            gsub("label\\\\_session", "individual", x)
        }
    )
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
