"""
Build contributions.xlsx for the Stata .do file to import.

The .do file at analysis/analysis/dynamic_regression.do runs:
  import excel ".../contributions.xlsx", sheet("contributions") firstrow

and expects columns: session_code, treatment, segment, round, participant_id,
contribution, payoff. We write a workbook with a single sheet named
"contributions", first row = variable names, all rows in the same order as the
source CSV so row numbering and row-based lags (contribution[_n-1]) are
preserved.

Author: Claude Code
Date: 2026-04-13
"""

from pathlib import Path

import pandas as pd
from openpyxl import Workbook

INPUT_CSV = Path(__file__).parent.parent / "datastore" / "derived" / "contributions.csv"
OUTPUT_XLSX = Path(__file__).parent.parent / "datastore" / "derived" / "contributions.xlsx"

# Variables the .do file actually references. Writing only these keeps the
# workbook small and avoids Stata choking on all-NA columns (e.g. `role`).
STATA_COLUMNS = [
    "session_code",
    "treatment",
    "segment",
    "round",
    "participant_id",
    "contribution",
    "payoff",
]


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    missing = [c for c in STATA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {INPUT_CSV}: {missing}")
    df = df[STATA_COLUMNS].copy()

    # Sanity: integer-typed columns should contain no NaNs; Stata reads a
    # blank cell as a missing value, which would silently drop rows.
    int_cols = ["treatment", "round", "participant_id"]
    for col in int_cols:
        if df[col].isna().any():
            raise ValueError(f"Column {col} has NaN values; Stata import would drop rows.")
        df[col] = df[col].astype(int)

    wb = Workbook()
    ws = wb.active
    ws.title = "contributions"
    ws.append(STATA_COLUMNS)
    for row in df.itertuples(index=False, name=None):
        ws.append(row)

    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUTPUT_XLSX)

    print(f"Wrote {len(df):,} rows to {OUTPUT_XLSX}")
    print(f"Sheet: 'contributions'")
    print(f"Columns: {STATA_COLUMNS}")
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
