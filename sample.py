import argparse
import pandas as pd
from cic_to_flowmeter import cic_to_pyflowmeter_columns
from config import class_names


def main():
    parser = argparse.ArgumentParser(description="Sample random rows from CSV files.")
    parser.add_argument("--from-csv-path", required=True, nargs="+", help="Paths to input CSV files")
    parser.add_argument("--to-csv-path", required=True, help="Path to the output CSV file")
    parser.add_argument("--count", required=True, type=int, help="Number of random rows to select per file")
    args = parser.parse_args()

    parts = []
    for path in args.from_csv_path:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()

        # Find the label column (before or after CIC rename)
        if "Label" in df.columns:
            label_col = "Label"
        elif "label" in df.columns:
            label_col = "label"
        else:
            print(f"WARNING: No label column found in {path}, skipping")
            continue

        # Filter to only rows with labels in class_names
        df[label_col] = df[label_col].astype(str).str.strip()
        df = df[df[label_col].isin(class_names)]

        n = min(args.count, len(df))
        sample = df.sample(n=n)
        parts.append(sample)
        print(f"Sampled {n} rows from {path} (filtered to {class_names})")

    combined = pd.concat(parts, ignore_index=True)
    combined.to_csv(args.to_csv_path, index=False)
    print(f"Wrote {len(combined)} total rows to {args.to_csv_path}")


if __name__ == "__main__":
    main()
