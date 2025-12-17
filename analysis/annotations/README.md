# Annotations Data Processing Pipeline

This directory contains scripts for processing experimental data from oTree PageTimes exports into annotation files suitable for behavioral analysis.

## Overview

The pipeline consists of two scripts that transform raw page timing data into structured annotations:

1. **`build_edited_data_csv.py`** - Transforms raw PageTimes CSV into normalized event data
2. **`generate_annotations.py`** - Creates annotation markers from the normalized event data

## Pipeline Flow

```
PageTimes CSV + Timesheet → build_edited_data_csv.py → edited_data_output.csv → generate_annotations.py → annotation_generated.csv
```

---

## Script 1: build_edited_data_csv.py

### Purpose
Transforms raw oTree PageTimes export data into a normalized format with timing calculations and participant-sequential ordering.

### Inputs
- **PageTimes CSV**: Raw oTree export (e.g., `PageTimes-2025-09-11.csv`)
- **Timesheet Excel**: Participant recording start times (`timesheet.xlsx`)

### Output
- **edited_data_output.csv**: Normalized event data with derived time columns

### What It Does
1. **Filters** to most recent experimental session
2. **Trims** each participant's data to their last `InitializeParticipant` event
3. **Reorders** from interleaved (by time) to sequential (by participant)
4. **Adds time columns**:
   - Excel serial dates for timestamps
   - Timezone adjustments (GMT → LOCAL → RECORDING)
   - Time elapsed since recording start (in days, seconds, milliseconds)

### Configuration

**Command-line Arguments (recommended):**

```bash
rye run python annotations/build_edited_data_csv.py \
  --pagetimes "/path/to/PageTimes-YYYY-MM-DD.csv" \
  --timesheet "/path/to/timesheet.xlsx" \
  --output "/path/to/edited_data_output.csv"
```

**Default Values:**
- `--pagetimes`: `/Users/caleb/Research/emotionsgame_5/analysis/datastore/annotations/PageTimes-2025-09-11.csv`
- `--timesheet`: `/Users/caleb/Research/emotionsgame_5/analysis/datastore/timesheet.xlsx`
- `--output`: `/Users/caleb/Research/emotionsgame_5/analysis/datastore/annotations/edited_data_output.csv`
- `TZ_NAME`: `"America/Chicago"` (hardcoded constant)
- `CDT_OFFSET_HOURS`: `5` (hardcoded constant)

### Usage

**Run with default paths:**

```bash
cd /Users/caleb/Research/emotionsgame_5/analysis
rye run python annotations/build_edited_data_csv.py
```

**Run with custom paths:**

```bash
cd /Users/caleb/Research/emotionsgame_5/analysis
rye run python annotations/build_edited_data_csv.py \
  --pagetimes "/custom/path/PageTimes-2025-10-24.csv" \
  --timesheet "/custom/path/timesheet.xlsx" \
  --output "/custom/path/output.csv"
```

### Output Columns
| Column | Description |
|--------|-------------|
| session_code | oTree session identifier |
| participant_id_in_session | Participant ID (1-16) |
| participant_code | oTree participant code |
| page_index | Sequential page index |
| app_name | oTree app name (e.g., supergame1) |
| page_name | Page name (e.g., Contribute, Results) |
| epoch_time_completed | Unix timestamp |
| date time | Excel serial date (UTC) |
| gmt | Excel serial date (GMT) |
| LOCAL | Excel serial date (local timezone) |
| RECORDING | Recording start time from timesheet |
| (unnamed) | Time since recording start (days) |
| (unnamed) | Time since recording start (seconds) |
| (unnamed) | Time since recording start (milliseconds) |

---

## Script 2: generate_annotations.py

### Purpose
Generates annotation markers from event data with duration filtering and occurrence counting.

### Input
- **edited_data_output.csv**: Output from `build_edited_data_csv.py`

### Output
- **annotation_generated.csv**: Annotation file with marker names and timing

### What It Does
1. **Maps participants** to respondent names (1→A1, 2→B1, ..., 16→R1, skipping I and O)
2. **Calculates durations**: Start/end times from successive page timestamps
3. **Filters events**: Only includes pages with duration > 1 second
4. **Generates markers**: Page names with occurrence counters within each supergame
   - Example: First "Contribute" in supergame1 → `Contribute_1`
   - Second "Contribute" in supergame1 → `Contribute_2`

### Configuration

**Command-line Arguments (recommended):**

```bash
rye run python annotations/generate_annotations.py \
  --input "/path/to/edited_data_output.csv" \
  --output "/path/to/annotation_generated.csv" \
  --duration-threshold 1000
```

**Default Values:**
- `--input`: `annotations/edited_data_output.csv` (script-relative path)
- `--output`: `annotations/annotation_generated.csv` (script-relative path)
- `--duration-threshold`: `1000` (milliseconds)

### Usage

**Run with default paths:**

```bash
cd /Users/caleb/Research/emotionsgame_5/analysis
rye run python annotations/generate_annotations.py
```

**Run with custom paths:**

```bash
cd /Users/caleb/Research/emotionsgame_5/analysis
rye run python annotations/generate_annotations.py \
  --input "/custom/path/edited_data_output.csv" \
  --output "/custom/path/my_annotations.csv" \
  --duration-threshold 500
```

**Run with just a different threshold:**

```bash
rye run python annotations/generate_annotations.py --duration-threshold 2000
```

### Output Format
The script produces a CSV with 8 columns matching the annotation schema:

| Column | Description | Example |
|--------|-------------|---------|
| Respondent Name | Participant identifier | A1, B1, C1, ... |
| Stimulus Name | (empty) | |
| Marker Type | Fixed value | Respondent Annotation |
| Marker Name | Page name with counter | Contribute_1, Results_2 |
| Start Time (ms) | Event start timestamp | 484000 |
| End Time (ms) | Event end timestamp | 786000 |
| Comment | (empty) | |
| Color | (empty) | |

### Participant Mapping
The script maps `participant_id_in_session` to letter codes:

```
1  → A1     9  → J1
2  → B1     10 → K1
3  → C1     11 → L1
4  → D1     12 → M1
5  → E1     13 → N1
6  → F1     14 → P1
7  → G1     15 → Q1
8  → H1     16 → R1
```
*(I and O are skipped to avoid confusion with 1 and 0)*

---

## Complete Workflow Example

### Step 1: Prepare Input Files
Place your files in the datastore:
```
emotionsgame_5/analysis/datastore/
├── annotations/
│   └── PageTimes-2025-09-11.csv
└── timesheet.xlsx
```

### Step 2: Build Edited Data
```bash
cd /Users/caleb/Research/emotionsgame_5/analysis
rye run python annotations/build_edited_data_csv.py
```

Expected output:
```
Loaded 2500 rows from PageTimes CSV
Loaded 16 participant recording start times
Filtered to most recent session: sa7mprty (2500 rows)
After trimming: 2400 rows total
Reordered data sequentially by participant
Added derived time columns (H-N)
Output written to: .../edited_data_output.csv
```

### Step 3: Generate Annotations
```bash
rye run python annotations/generate_annotations.py
```

Expected output:
```
Loading data from: .../edited_data_output.csv
Loaded 2400 rows
Detected time column: (last unnamed column)
After dropping nulls: 2400 rows
After filtering (duration > 1000ms): 856 rows

Sample output (first 5 rows):
  Respondent Name Stimulus Name        Marker Type  ...
0              A1                Respondent Annotation  ...
...

Output written to: .../annotation_generated.csv
Total rows: 856
```

---

## Troubleshooting

### Script 1 Issues

**"Missing required columns" error**
- Verify PageTimes CSV has columns: `session_code`, `participant_id_in_session`, `participant_code`, `page_index`, `app_name`, `page_name`, `epoch_time_completed`

**"Could not parse timesheet time" warnings**
- Check that timesheet.xlsx columns D (participant) and E (start time) are properly formatted
- Ensure participant letters match: A, B, C, D, E, F, G, H, J, K, L, M, N, P, Q, R

**Incorrect timezone calculations**
- Adjust `TZ_NAME` and `CDT_OFFSET_HOURS` constants for your experiment location

### Script 2 Issues

**"No numeric columns found for timestamp" error**
- Ensure edited_data_output.csv has the unnamed millisecond column (last column)
- Check that the rightmost column contains numeric values

**Too few or too many annotations**
- Adjust `DURATION_THRESHOLD_MS` to change filtering sensitivity
- Default is 1000ms (1 second); lower values include more brief page transitions

**"participant_id_in_session must be 1-16" error**
- Verify input data only contains participant IDs in valid range
- Check for data corruption or unexpected participant IDs

---

## Output File Locations

Default file paths (configured in scripts):

```
/Users/caleb/Research/emotionsgame_5/analysis/datastore/
├── annotations/
│   ├── PageTimes-2025-09-11.csv          [input to script 1]
│   ├── edited_data_output.csv            [output of script 1, input to script 2]
│   └── annotation_generated.csv          [output of script 2]
└── timesheet.xlsx                         [input to script 1]
```

---

## Notes

- Both scripts follow the project's Python style guidelines (max 5 indentation levels, functions separated logically)
- Scripts use `rye` for Python environment management
- Time calculations assume Unix epoch timestamps and Excel serial date format
- The annotation format is compatible with behavioral analysis software expecting this schema
