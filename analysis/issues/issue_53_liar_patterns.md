# Issue #53: Liar Patterns — Buckets, Conditional Probability, and Behavioral Correlates

## Overview

Investigate patterns of deceptive behavior (lying) across the experiment. This issue has three components: classifying participants into liar buckets by frequency, estimating the conditional probability of repeated lying via logistic regression, and comparing sentiment and facial expression signatures across liar buckets.

## Definitions

### Liar Threshold
A player "lied" in a given round if they made a promise and contributed below 20 (`lied_this_round_20 == True`). This uses the high threshold from issue #6.

### Liar Buckets
Participants are classified by total lying rounds across the entire experiment (all supergames):

| Bucket | Lie Count | Description |
|--------|-----------|-------------|
| never | 0 | Never broke a promise |
| one_time | 1 | Broke a promise exactly once |
| moderate | 2-3 | Occasional promise-breaking |
| severe | 4+ | Frequent promise-breaking |

### Conditional Probability
P(liar_t | liar_{t-1}) — the probability of lying in the current round given that the player lied in the previous round, estimated via logistic regression with controls.

## Scripts

### 1. `analysis/derived/liar_buckets.py` — Bucket Classification

**Input**: `analysis/datastore/derived/behavior_classifications.csv` (3,520 rows)

**Logic**:
1. Count rounds where `lied_this_round_20 == True` per participant across all sessions and supergames
2. Assign bucket labels based on count thresholds (0 / 1 / 2-3 / 4+)
3. Validate: exactly 160 unique participants, all assigned a bucket

**Output**: `analysis/datastore/derived/liar_buckets.csv` (160 rows, one per participant)

| Column | Type | Description |
|--------|------|-------------|
| session_code | string | Session identifier |
| treatment | int | Treatment group (1 or 2) |
| label | string | Player label (A-R, skipping I/O) |
| participant_id | int | Participant ID |
| lie_count | int | Total rounds with `lied_this_round_20 == True` |
| liar_bucket | string | Bucket category: never, one_time, moderate, severe |

### 2. `analysis/analysis/issue_53_liar_regression.R` — Conditional Probability Regression

**Inputs**:
- `analysis/datastore/derived/behavior_classifications.csv` — liar flags
- Raw session CSVs in `analysis/datastore/` — gender extraction (`finalresults.1.player.q1`)

**Model**:
```r
feglm(lied_this_round_20 ~ lied_prev_round + gender + treatment | segment,
      family = binomial, cluster = ~label_session, data = df)
```

**Steps**:
1. Load behavior classifications and extract gender from raw session CSVs
2. Merge gender onto behavior data by session_code + label
3. Create lagged `lied_this_round_20` within each participant x segment
4. Drop first round of each segment (no lag available; ~2,720 obs remain)
5. Estimate logistic regression with segment fixed effects and participant-clustered SEs
6. Export LaTeX regression table

**Output**: `analysis/output/tables/liar_conditional_probability.tex`

### 3. `analysis/issues/issue_53_liar_plots.py` — Bucket-Level Visualizations

**Inputs**:
- `analysis/datastore/derived/merged_panel.csv` — sentiment and emotion data at player-round level
- `analysis/datastore/derived/liar_buckets.csv` — bucket classification

**Steps**:
1. Merge liar buckets onto merged panel by session_code + label
2. Filter to `page_type == "Contribute"` (one row per player-round)
3. Drop round 1 of each segment (no prior chat to generate sentiment)

**Outputs**:
- `analysis/output/plots/sentiment_by_liar_bucket.png` — Box plot of `sentiment_compound_mean` by liar bucket, with n counts per bucket
- `analysis/output/plots/emotions_by_liar_bucket.png` — Faceted box plots of anger, contempt, joy, sadness, and surprise by liar bucket
- `analysis/output/summary_statistics/liar_bucket_summary.tex` — LaTeX table with means and SDs by bucket for sentiment and all emotions

## Key Design Decisions

- **Threshold**: 20 only (contribution < 20 after promise). The low threshold (< 5) was not used for this analysis.
- **Bucket scope**: Across the entire experiment, not per-supergame. A participant's bucket reflects their total lying frequency.
- **Regression framework**: R/fixest for logistic regression, consistent with other regression scripts in the project (issues #46, #49).
- **Visualization style**: Box plots. Simple and focused.
- **Emotion subset for plots**: anger, contempt, joy, sadness, surprise (5 of 13 available emotion columns — selected for theoretical relevance to deception).

## Dependencies

- Issue #2: Promise Classification — provides `promise_classifications.csv`
- Issue #6: Behavior Classification — provides `behavior_classifications.csv` with `lied_this_round_20` column
- Issue #38: Merge Panel Data — provides `merged_panel.csv` with sentiment and emotion columns

## Testing

Python scripts are covered by a shared test file:
- `analysis/tests/test_liar_buckets.py` — tests for liar bucket classification (Task 1) and visualization outputs (Task 3)
