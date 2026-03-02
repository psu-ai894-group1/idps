import argparse
import os
import time
import dotenv
dotenv.load_dotenv()

from pyflowmeter.sniffer import create_sniffer


def main():
    parser = argparse.ArgumentParser(description="Capture network flows to a CSV file.")
    parser.add_argument('-i', '--iface', default='eth0', help='Network interface to capture (default: eth0)')
    parser.add_argument("--to-csv-path", required=True, help="Path to save captured flows")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.to_csv_path) or ".", exist_ok=True)

    sniffer = create_sniffer(
        input_interface=args.iface,
        to_csv=True,
        output_file=args.to_csv_path,
    )

    print(f"Capturing flows on {args.iface} -> {args.to_csv_path}")
    print("Press Ctrl-C to stop.")
    sniffer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sniffer.stop()
        print(f"\nStopped. Flows saved to {args.to_csv_path}")


if __name__ == "__main__":
    main()
