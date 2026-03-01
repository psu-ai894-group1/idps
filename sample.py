import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Sample random rows from a CSV file.")
    parser.add_argument("--from-csv-path", required=True, help="Path to the input CSV file")
    parser.add_argument("--to-csv-path", required=True, help="Path to the output CSV file")
    parser.add_argument("--count", required=True, type=int, help="Number of random rows to select")
    args = parser.parse_args()

    df = pd.read_csv(args.from_csv_path)
    sample = df.sample(n=args.count)
    sample.to_csv(args.to_csv_path, index=False)
    print(f"Sampled {args.count} rows from {args.from_csv_path} -> {args.to_csv_path}")

if __name__ == "__main__":
    main()
