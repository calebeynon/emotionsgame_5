# Issue #33: Add Comprehensive Summary Statistics

**Status:** Complete
**Branch:** issue_33_summary_statistics
**Created:** 2026-03-02

## Objective

Compute and store a comprehensive set of descriptive statistics from the experiment data in a single, organized output location (`analysis/output/summary_statistics/`). The goal is a reference library so that when someone asks for a specific statistic, you know exactly where to find it.

## Architecture

A modular pipeline of 8 domain-specific scripts sharing common utilities, orchestrated by `run_all.py`:

```
analysis/analysis/summary_statistics/
  run_all.py          # Orchestrator
  ss_common.py        # Shared data loading, LaTeX formatting, output utilities
  ss_contributions.py # Contribution distributions
  ss_chat.py          # Chat & communication
  ss_sentiment.py     # Sentiment analysis
  ss_behavior.py      # Promise & behavioral classification
  ss_payoffs.py       # Payoff distributions
  ss_groups.py        # Group dynamics
  ss_demographics.py  # Demographics & survey
  ss_experiment_totals.py  # Experiment-level totals
```

## Inputs

| File | Description |
|------|-------------|
| `datastore/raw/all_apps_wide_*.csv` | Raw oTree data CSVs (one per session) |
| `datastore/raw/chat_messages_*.csv` | Raw oTree chat CSVs (one per session) |
| `datastore/derived/behavior_classifications.csv` | Player-round liar/sucker/promise flags |
| `datastore/derived/sentiment_scores.csv` | Player-round VADER sentiment scores |
| `datastore/raw/demographics.csv` | Participant demographics survey data |

All data is loaded via `ss_common.load_experiment()` and `ss_common.load_contributions_df()` which use the `experiment_data.py` module.

## Outputs

All outputs land in `analysis/output/summary_statistics/`:

### Contributions (`ss_contributions.py`)
| File | Description |
|------|-------------|
| `contributions_descriptive.tex` | Mean, median, SD, skewness, kurtosis by treatment |
| `contributions_frequencies.tex` | Contribution frequencies (0, 1-12, 13-24, 25) by treatment |
| `contributions_extremes.tex` | % zero and % max contributions by supergame, round, treatment |
| `contributions_histogram_by_supergame.png` | Histogram faceted by supergame |
| `contributions_histogram_by_treatment.png` | Histogram faceted by treatment |

### Chat (`ss_chat.py`)
| File | Description |
|------|-------------|
| `chat_volume.tex` | Total messages, mean per player per round by treatment/supergame |
| `chat_length.tex` | Mean message length (chars/words) by treatment/supergame |
| `chat_participation.tex` | Chat participation rate by treatment/supergame |
| `chat_orphan_volume.tex` | Orphan chat volume by treatment/supergame |
| `chat_word_frequency.tex` | Top 20 most frequent words |

### Sentiment (`ss_sentiment.py`)
| File | Description |
|------|-------------|
| `sentiment_descriptive.tex` | Mean compound sentiment by treatment/supergame |
| `sentiment_components.tex` | Positive/negative/neutral component means |
| `sentiment_categories.tex` | % positive/negative/neutral messages |
| `sentiment_intensity.tex` | Intensity distribution (strong/moderate/mild/neutral) |
| `sentiment_contribution_correlation.tex` | Sentiment-contribution correlation |

### Behavior (`ss_behavior.py`)
| File | Description |
|------|-------------|
| `behavior_promise_rates.tex` | Promise rates by treatment x round |
| `behavior_liar_rates.tex` | Liar rates (both thresholds) by treatment/supergame |
| `behavior_sucker_rates.tex` | Sucker rates by treatment/supergame |
| `behavior_persistence.tex` | Behavioral type persistence across rounds |
| `behavior_conditional_contribution.tex` | Mean contribution given promise vs no-promise |

### Payoffs (`ss_payoffs.py`)
| File | Description |
|------|-------------|
| `payoffs_summary.tex` | Mean/median total payoff by treatment |
| `payoffs_by_supergame.tex` | Payoff breakdown per supergame |
| `payoffs_inequality.tex` | Gini coefficient by treatment |
| `payoffs_dollar_distribution.tex` | Final dollar earnings distribution |

### Groups (`ss_groups.py`)
| File | Description |
|------|-------------|
| `groups_cooperation.tex` | Group cooperation rate by treatment/supergame |
| `groups_free_riders.tex` | Free-rider count per group by treatment/supergame |
| `groups_within_sd.tex` | Within-group contribution SD by treatment/supergame |
| `groups_regrouping_effect.tex` | Contribution change at supergame boundaries |

### Demographics (`ss_demographics.py`)
| File | Description |
|------|-------------|
| `demographics_gender.tex` | Gender distribution by treatment |
| `demographics_age.tex` | Age summary by treatment |
| `demographics_ethnicity.tex` | Ethnicity distribution by treatment |
| `demographics_religion.tex` | Religion importance distribution by treatment |
| `demographics_siblings.tex` | Siblings summary by treatment |
| `demographics_contribution_correlation.tex` | Demographic-contribution correlations |

### Experiment Totals (`ss_experiment_totals.py`)
| File | Description |
|------|-------------|
| `experiment_totals.tex` | Total N, sessions, groups, messages, observations |
| `experiment_timing.tex` | Average experiment timing from PageTimes |

### Review
| File | Description |
|------|-------------|
| `review_all_tables.tex` | LaTeX document compiling all tables |
| `review_all_tables.pdf` | Compiled PDF for quick review |

## Testing

| File | Description | Script |
|------|-------------|--------|
| `tests/test_summary_statistics.py` | Unit tests for utilities and module execution | `pytest` |
| `tests/verify_data_pipeline.py` | End-to-end data tracing from raw CSV to tables | standalone |

## Key Design Decisions

- **Sample SD (ddof=1)**: All standard deviations and coefficients of variation use sample SD
- **Sentiment correlation**: Uses Spearman rank correlation (non-normal sentiment data)
- **Gini coefficient**: Computed on per-supergame payoffs, not cumulative totals
- **Demographics merge**: Joins on `participant.label` to match experiment_data player labels
- **Contribution frequencies**: Denominator is total observations per treatment, not per bin
