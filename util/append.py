import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
from cic_to_flowmeter import cic_to_pyflowmeter_columns

def normalize_columns(df):
    """Normalize column names: apply CIC-to-pyflowmeter mapping, then lowercase."""
    df = cic_to_pyflowmeter_columns(df)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_')
    return df

def main():
    parser = argparse.ArgumentParser(description="Append one CSV to another.")
    parser.add_argument("--source", required=True, help="CSV file to read from")
    parser.add_argument("--dest", required=True, help="CSV file to append to")
    args = parser.parse_args()

    source_df = normalize_columns(pd.read_csv(args.source))
    dest_df = normalize_columns(pd.read_csv(args.dest))

    combined = pd.concat([dest_df, source_df], ignore_index=True)
    combined.to_csv(args.dest, index=False)

    print(f"Appended {len(source_df)} rows from {args.source} to {args.dest}")
    print(f"Total rows: {len(combined)}")

if __name__ == "__main__":
    main()
