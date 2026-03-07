import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Filter flows CSV by IP, port, and row count.")
    parser.add_argument("--csv-path", required=True, help="Path to the flows CSV file")
    parser.add_argument("--ip", default=None, help="IP address to match (src or dst)")
    parser.add_argument("--port", type=int, default=None, help="Port to match (src or dst)")
    parser.add_argument("--count", type=int, default=None, help="Max rows to return")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_')

    if args.ip is not None:
        df = df[(df['src_ip'] == args.ip) | (df['dst_ip'] == args.ip)]

    if args.port is not None:
        df = df[(df['src_port'] == args.port) | (df['dst_port'] == args.port)]

    if args.count is not None:
        df = df.head(args.count)

    print(df.to_csv(index=False), end='')

if __name__ == "__main__":
    main()
