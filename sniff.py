"""
AI-894
This file captures network traffic and extracts flow features using pyflowmeter. It can save the extracted flows to a CSV file for analysis.
Authors: Karla Gonzalez Caballero (kxg5613@psu.edu), Christopher Umbel (czu5008@psu.edu)
"""
import argparse
import logging
import os
import time
import dotenv
dotenv.load_dotenv()

from pyflowmeter.sniffer import create_sniffer
from pyflowmeter.features.flow_bytes import FlowBytes
from pyflowmeter.features.context.packet_direction import PacketDirection

def _patched_get_min_forward_header_bytes(self):
    packets = self.feature.packets
    if not packets:
        return 0
    forward_sizes = [
        self._header_size(packet)
        for packet, direction in packets
        if direction == PacketDirection.FORWARD
    ]
    return min(forward_sizes) if forward_sizes else 0


FlowBytes.get_min_forward_header_bytes = _patched_get_min_forward_header_bytes

# Monkey-patch for pyflowmeter bug: ZeroDivisionError in get_bulk_rate
# when a flow has no backward packets.
_original_get_bulk_rate = FlowBytes.get_bulk_rate

def _patched_get_bulk_rate(self, direction):
    try:
        return _original_get_bulk_rate(self, direction)
    except ZeroDivisionError:
        logging.error(f"ZeroDivisionError in get_bulk_rate for flow {self.flow_id} {direction}")
        return 0

FlowBytes.get_bulk_rate = _patched_get_bulk_rate

def setup_logging():
    """
    Configure logging to output/sniff.log and stdout.
    """
    log_dir = "output"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "sniff.log")
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    for h in root.handlers[:]:
        root.removeHandler(h)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    root.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    root.addHandler(stream_handler)

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Capture network flows to a CSV file.")
    parser.add_argument('-i', '--iface', default='eth0', help='Network interface to capture (default: eth0)')
    parser.add_argument("--to-csv-path", required=True, help="Path to save captured flows")
    args = parser.parse_args()

    # Tune pyflowmeter timeouts to produce shorter flows similar to CICFlowMeter.
    # Lower idle timeout so flows are finalized sooner.
    import pyflowmeter.flow_session as _fs
    _fs.EXPIRED_UPDATE = 5
    _fs.FlowSession.GARBAGE_COLLECT_PACKETS = 100

    os.makedirs(os.path.dirname(args.to_csv_path) or ".", exist_ok=True)

    sniffer = create_sniffer(
        input_interface=args.iface,
        to_csv=True,
        output_file=args.to_csv_path,
    )

    logging.info("Capturing flows on %s -> %s", args.iface, args.to_csv_path)
    logging.info("Press Ctrl-C to stop.")
    sniffer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sniffer.stop()
        logging.info("Stopped. Flows saved to %s", args.to_csv_path)

if __name__ == "__main__":
    main()
