import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Count flows matching an IP and port filter.")
    parser.add_argument("--csv-path", required=True, help="Path to the flows CSV file")
    parser.add_argument("--ip", required=True, help="IP address to match (src or dst)")
    parser.add_argument("--port", type=int, default=None, help="Port to match (src or dst)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_')

    mask = (df['src_ip'] == args.ip) | (df['dst_ip'] == args.ip)
    if args.port is not None:
        mask = mask & ((df['src_port'] == args.port) | (df['dst_port'] == args.port))
    print(f"Total rows: {len(df)}")
    print(f"Matched: {mask.sum()}")

if __name__ == "__main__":
    main()
