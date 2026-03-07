import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Apply log1p transform to all numeric columns.")
    parser.add_argument("--from-csv", required=True, help="Path to the input CSV file")
    args = parser.parse_args()

    df = pd.read_csv(args.from_csv)
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = np.log1p(np.abs(df[numeric_cols])) * np.sign(df[numeric_cols])

    print(df.to_csv(index=False), end='')

if __name__ == "__main__":
    main()
