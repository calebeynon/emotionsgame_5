# Multi-Session Experimental Data Analysis

## Overview

This document describes the `multi_session_analysis.py` script, which provides comprehensive analysis and visualization for experimental data containing multiple sessions. The script extends the original single-session analysis to handle experiments with multiple sessions, providing both aggregate (cross-session) and individual session-level analysis.

## Architecture

### Data Structure Hierarchy
```
Experiment
├── Session 1 (Treatment A)
│   ├── Segment (supergame1, supergame2, etc.)
│   │   ├── Round 1, 2, 3...
│   │   │   ├── Group 1, 2, 3, 4
│   │   │   │   └── Player A, B, C, D
│   │   │   │       ├── contribution
│   │   │   │       ├── payoff  
│   │   │   │       └── chat_messages
│   │   │   └── chat_messages (round-level)
│   │   └── chat_messages (segment-level)
│   └── finalresults (survey data)
└── Session 2 (Treatment B)
    └── ... (same structure)
```

## Key Features

### 1. Dual Analysis Framework

#### **Segment-by-Segment Analysis (Cross-Session Aggregation)**
- Aggregates data across all sessions to show overall patterns
- Provides statistical measures (means, standard errors) across sessions
- Shows treatment effects when multiple treatments are present
- Useful for understanding general experimental patterns

#### **Session-by-Session Analysis**
- Individual analysis for each session
- Allows comparison between sessions
- Identifies session-specific patterns or outliers
- Useful for quality control and detailed examination

### 2. Analysis Types

#### **Contribution Analysis**
- Average contributions by supergame (aggregate and per-session)
- Round-by-round contribution patterns within supergames
- Individual player contribution trends across supergames
- Statistical significance testing across sessions

#### **Ranking Analysis**
- Player rankings by total payoff
- Player rankings by average contribution  
- Cross-session aggregated rankings
- Individual session rankings

#### **Survey Analysis**
- Demographic distributions (age, gender, major, etc.)
- Treatment and session distributions
- Aggregated across all sessions

#### **Chat & Communication Analysis**
- Word frequency analysis across all messages
- Chat activity patterns by round and supergame
- Message volume trends

#### **Sentiment Analysis**
- Distribution of sentiment scores across all messages
- Sentiment trends by supergame
- Positive/negative/neutral message categorization

## Output Structure

```
multi_session_plots/
├── segment_analysis/                              # Cross-session aggregated plots
│   ├── aggregate_contributions_by_segment.png     # Bar chart with error bars
│   ├── contributions_by_round_aggregate.png       # Round-by-round patterns
│   ├── individual_trends_aggregate.png            # Player trends with CI
│   ├── payoff_rankings_aggregate.png              # Average payoff rankings
│   ├── contribution_rankings_aggregate.png        # Average contribution rankings
│   ├── survey_responses_aggregate.png             # Demographics across sessions
│   ├── chat_word_frequency_aggregate.png          # Most common words
│   ├── chat_messages_per_round_aggregate.png      # Communication patterns
│   └── sentiment_analysis_aggregate.png           # Sentiment distribution
├── session_analysis/                              # Individual session plots  
│   ├── contributions_by_session.png               # Grid of session comparisons
│   └── individual_trends_[SESSION_CODE].png       # One file per session
└── summary_statistics.txt                         # Comprehensive statistics
```

## Installation & Setup

### Prerequisites
```bash
# Required Python packages
pip install pandas numpy matplotlib seaborn scipy nltk
```

### Data Requirements
The script expects data in the format produced by the `experiment_data.py` module:
- **Data CSV files**: Experimental data with contribution, payoff, and demographic information
- **Chat CSV files**: Chat messages with timestamps and sentiment scores (optional)
- **Treatment codes**: Integer treatment identifiers (1, 2, etc.)

## Usage

### Basic Usage
1. **Configure data paths** in the `main()` function:
```python
file_pairs = [
    ('/path/to/session1_data.csv', '/path/to/session1_chat.csv', 1),  # Treatment 1
    ('/path/to/session2_data.csv', '/path/to/session2_chat.csv', 2),  # Treatment 2
    ('/path/to/session3_data.csv', '/path/to/session3_chat.csv', 1),  # Treatment 1
    # Add more sessions as needed
]
```

2. **Run the script**:
```bash
cd /path/to/analysis/directory
python multi_session_analysis.py
```

### Advanced Configuration
- **Output directory**: Modify `DEFAULT_OUTPUT_DIR` constant
- **Color schemes**: Adjust `COLORS` and `SESSION_COLORS` arrays
- **Statistical thresholds**: Modify significance levels in individual functions
- **Plot dimensions**: Adjust `plt.rcParams['figure.figsize']` globally

## Statistical Methods

### Cross-Session Aggregation
- **Standard Error**: SE = σ/√n, where n is number of sessions
- **Error Bars**: Display ±1 standard error for cross-session means
- **Box Plots**: Show distribution of session-level means for each condition

### Individual Session Analysis  
- **Standard Deviation**: Within-session variability
- **Treatment Effects**: Side-by-side comparison of sessions with different treatments
- **Quality Control**: Identify outlier sessions or anomalous patterns

## Function Reference

### Data Extraction Functions
- `extract_contributions_data(experiment)`: Extracts all contribution/payoff data into DataFrame
- `extract_chat_data(experiment)`: Extracts all chat messages with sentiment scores
- `extract_survey_data(experiment)`: Extracts demographic survey responses

### Visualization Functions
- `plot_contributions_aggregate_by_segment()`: Cross-session contribution analysis
- `plot_contributions_by_session()`: Individual session contribution analysis  
- `plot_individual_trends_aggregate()`: Player behavior trends (aggregated)
- `plot_individual_trends_by_session()`: Player behavior trends (per session)
- `plot_payoff_rankings_aggregate()`: Cross-session payoff rankings
- `plot_contribution_rankings_aggregate()`: Cross-session contribution rankings
- `analyze_survey_responses_aggregate()`: Demographic analysis
- `generate_chat_word_frequency_aggregate()`: Communication word analysis
- `plot_chat_messages_per_round_aggregate()`: Communication volume analysis
- `plot_sentiment_analysis_aggregate()`: Sentiment pattern analysis
- `generate_summary_statistics()`: Comprehensive numerical summaries

### Utility Functions
- `setup_output_directories()`: Creates organized output folder structure

## Comparison with Original Analysis

### Enhanced Features
| Original Script | Multi-Session Script |
|----------------|---------------------|
| Single session only | Multiple sessions with aggregation |
| Basic error bars (within-session) | Statistical error bars (across-sessions) |
| Individual plots only | Both aggregate and individual analyses |
| Single output directory | Organized output structure |
| Treatment blind | Treatment-aware comparisons |
| Limited statistical testing | Comprehensive statistical summaries |

### Maintained Features
- Visual style and color schemes
- Plot types and layouts
- Data processing pipelines
- Chat and sentiment analysis methods

## Best Practices

### Data Quality
1. **Verify data consistency** across sessions (same number of rounds, players, etc.)
2. **Check treatment coding** is consistent and meaningful
3. **Validate chat data alignment** with experimental sessions
4. **Review summary statistics** for outliers or data quality issues

### Interpretation Guidelines
1. **Focus on aggregate patterns** for general conclusions
2. **Use individual session analysis** for quality control and detailed investigation
3. **Compare error bars** to assess statistical significance of differences
4. **Consider treatment effects** when interpreting cross-session patterns

### Performance Considerations
- **Large datasets**: Consider sampling for chat word frequency analysis
- **Memory usage**: Process sessions individually if memory constraints exist
- **Plot resolution**: Adjust DPI settings based on intended use (screen vs. print)

## Troubleshooting

### Common Issues
1. **"No data files specified"**: Update `file_pairs` in `main()` function
2. **Missing packages**: Install required dependencies with pip
3. **Empty plots**: Check data file paths and formats
4. **Memory errors**: Reduce number of sessions or increase system memory
5. **Font errors**: Install required system fonts or modify font settings

### Debug Mode
Add debugging prints by modifying the script:
```python
# Add after data extraction
print(f"Contribution data shape: {contrib_df.shape}")
print(f"Sessions found: {contrib_df['session_code'].unique()}")
print(f"Treatments found: {contrib_df['treatment'].unique()}")
```

## Version History
- **v1.0**: Initial multi-session analysis framework
- Compatible with `experiment_data.py` v1.0
- Extends `analysis_plots.py` functionality

## Contact
For questions about this analysis framework, refer to the experiment documentation or the original `analysis_plots.py` script for single-session analysis patterns.