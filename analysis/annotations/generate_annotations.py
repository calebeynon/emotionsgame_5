"""
Generate annotation CSV from edited_data_output.csv

Transforms page event data into annotation format with:
- Respondent name mapping (participant_id_in_session -> A1, B1, etc.)
- Marker names with occurrence counters within supergames
- Start/end times calculated from successive page timestamps
- Filtering for events > 1 second duration
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


# Global constants
INPUT_DIR = Path(__file__).parent
INPUT_FILENAME = "edited_data_output.csv"
OUTPUT_FILENAME = "annotation_generated.csv"
DURATION_THRESHOLD_MS = 1000
RESPONDENT_SUFFIX = "1"
RESPONDENT_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R"]


def load_data(path):
    """Load CSV data with validation."""
    df = pd.read_csv(path)
    required_cols = ['participant_id_in_session', 'app_name', 'page_name']
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df


def detect_time_column(df):
    """Detect the rightmost numeric column as timestamp column."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        raise ValueError("No numeric columns found for timestamp")
    
    time_col = numeric_cols[-1]
    print(f"Detected time column: '{time_col}'")
    
    return time_col


def id_to_respondent(pid):
    """Map participant_id_in_session to respondent name (A1, B1, ..., R1)."""
    if pid < 1 or pid > 16:
        raise ValueError(f"participant_id_in_session must be 1-16, got {pid}")
    
    return RESPONDENT_LABELS[pid - 1] + RESPONDENT_SUFFIX


def compute_durations(df, time_col):
    """Compute start/end times and durations for each page event."""
    df = df.sort_values(['participant_id_in_session', time_col]).reset_index(drop=True)
    
    df['start_time_ms'] = df[time_col].astype(int)
    df['end_time_ms'] = df.groupby('participant_id_in_session')[time_col].shift(-1)
    df['end_time_ms'] = df['end_time_ms'].fillna(df['start_time_ms']).astype(int)
    df['duration_ms'] = df['end_time_ms'] - df['start_time_ms']
    
    return df


def build_marker_names(df):
    """Build marker names with occurrence counters within each supergame."""
    df['occurrence'] = df.groupby(['participant_id_in_session', 'app_name', 'page_name']).cumcount() + 1
    df['marker_name'] = df['page_name'] + '_' + df['occurrence'].astype(str)
    
    return df


def assemble_output(df):
    """Create output DataFrame with required schema."""
    output = pd.DataFrame({
        'Respondent Name': df['respondent_name'],
        'Stimulus Name': '',
        'Marker Type': 'Respondent Annotation',
        'Marker Name': df['marker_name'],
        'Start Time (ms)': df['start_time_ms'],
        'End Time (ms)': df['end_time_ms'],
        'Comment': '',
        'Color': ''
    })
    
    output = output.sort_values(['Respondent Name', 'Start Time (ms)']).reset_index(drop=True)
    
    return output


def write_output_csv(df, path):
    """Write output CSV file."""
    df.to_csv(path, index=False)
    print(f"\nOutput written to: {path}")
    print(f"Total rows: {len(df)}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Generate annotations from edited data output')
    parser.add_argument('--input', default=INPUT_FILENAME, help='Input CSV filename')
    parser.add_argument('--output', default=OUTPUT_FILENAME, help='Output CSV filename')
    args = parser.parse_args()
    
    input_path = INPUT_DIR / args.input
    output_path = INPUT_DIR / args.output
    
    print(f"Loading data from: {input_path}")
    df = load_data(input_path)
    print(f"Loaded {len(df)} rows")
    
    time_col = detect_time_column(df)
    
    df = df.dropna(subset=['participant_id_in_session', 'app_name', 'page_name', time_col])
    print(f"After dropping nulls: {len(df)} rows")
    
    df['respondent_name'] = df['participant_id_in_session'].apply(id_to_respondent)
    
    df = compute_durations(df, time_col)
    
    df = build_marker_names(df)
    
    df_filtered = df[df['duration_ms'] > DURATION_THRESHOLD_MS].copy()
    print(f"After filtering (duration > {DURATION_THRESHOLD_MS}ms): {len(df_filtered)} rows")
    
    output = assemble_output(df_filtered)
    
    print("\nSample output (first 5 rows):")
    print(output.head())
    
    write_output_csv(output, output_path)


if __name__ == "__main__":
    main()
