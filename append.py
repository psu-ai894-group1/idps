import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Append one CSV to another.")
    parser.add_argument("--source", required=True, help="CSV file to read from")
    parser.add_argument("--dest", required=True, help="CSV file to append to")
    args = parser.parse_args()

    source_df = pd.read_csv(args.source)
    dest_df = pd.read_csv(args.dest)

    combined = pd.concat([dest_df, source_df], ignore_index=True)
    combined.to_csv(args.dest, index=False)

    print(f"Appended {len(source_df)} rows from {args.source} to {args.dest}")
    print(f"Total rows: {len(combined)}")

if __name__ == "__main__":
    main()
