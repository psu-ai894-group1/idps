import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Add or overwrite a label column in a CSV.")
    parser.add_argument("--csv-path", required=True, help="Path to the CSV file")
    parser.add_argument("--label", required=True, help="Label value to set for every row")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df['label'] = args.label
    df.to_csv(args.csv_path, index=False)

    print(f"Set label to '{args.label}' for {len(df)} rows in {args.csv_path}")

if __name__ == "__main__":
    main()
