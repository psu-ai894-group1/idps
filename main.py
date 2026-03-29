import argparse
import logging
import os
import dotenv
dotenv.load_dotenv()

from pyflowmeter.sniffer import create_sniffer
from pyflowmeter.features.flow_bytes import FlowBytes
from pyflowmeter.features.context.packet_direction import PacketDirection

# Monkey-patch for pyflowmeter bug: ValueError when a flow has no forward packets.
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
        return 0

FlowBytes.get_bulk_rate = _patched_get_bulk_rate
import netifaces
import time
import joblib
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from flows_to_tensors import flows_to_tensors
from config import class_names

import tensorflow as tf
from model import predict, load_model_weights
from db import get_connection, save_flows_and_inferences, set_data_path, get_db_path

MIN_INFERENCE_BATCH = 50

class FlowFileHandler(FileSystemEventHandler):
    def __init__(self, model, scaler, trace_path=None):
        self.model = model
        self.scaler = scaler
        self.byte_offset = 0
        self.header = None
        self.buffer = pd.DataFrame()
        self.trace_path = trace_path
        self.trace_header_written = False
        logging.info("FlowFileHandler initialized")

    def process_flows(self, flows_df):
        logging.info("Processing flows: %d", len(flows_df))
        node_features, adjacency, _, _ = flows_to_tensors(flows_df, scaler=self.scaler, log_transform=True)
        preds, confidences = predict(self.model, node_features, adjacency)

        for i in range(len(flows_df)):
            label = class_names[preds[i]]
            logging.info("Row %d predicted label: %s (confidence: %.4f)", i, label, confidences[i])

        save_flows_and_inferences(flows_df, preds, confidences, class_names)

        if self.trace_path:
            self._write_trace(node_features, preds, confidences, flows_df)

    def _write_trace(self, node_features, preds, confidences, flows_df):
        from config import feature_names, use_features
        available = [f for f in feature_names if f in use_features]
        features_np = node_features.numpy() if hasattr(node_features, 'numpy') else node_features
        trace_df = pd.DataFrame(features_np, columns=available)
        if 'dst_port' in trace_df.columns:
            trace_df['dst_port'] = flows_df['dst_port'].values
        else:
            trace_df.insert(0, 'dst_port', flows_df['dst_port'].values)
        trace_df.insert(0, 'dst_ip', flows_df['dst_ip'].values)
        if 'src_port' in flows_df.columns:
            trace_df.insert(0, 'src_port', flows_df['src_port'].values)
        trace_df.insert(0, 'src_ip', flows_df['src_ip'].values)
        trace_df['predicted_label'] = [class_names[p] for p in preds]
        trace_df['confidence'] = confidences

        write_header = not self.trace_header_written
        trace_df.to_csv(self.trace_path, mode='a', index=False, header=write_header)
        self.trace_header_written = True

    def _read_new_rows(self, path):
        """Read only the bytes appended since the last call."""
        with open(path, 'r') as f:
            # On first call, read and store the header line
            if self.header is None:
                self.header = f.readline()
                self.byte_offset = f.tell()

            f.seek(self.byte_offset)
            new_data = f.read()
            if not new_data or not new_data.strip():
                return pd.DataFrame()

            self.byte_offset = f.tell()

        from io import StringIO
        return pd.read_csv(StringIO(self.header + new_data))

    def on_modified(self, event):
        if event.src_path.endswith('.csv') and os.path.exists(event.src_path) and os.path.getsize(event.src_path) > 0:
            new_flows_df = self._read_new_rows(event.src_path)

            if len(new_flows_df) > 0:
                logging.info("New flows detected: %s (%d rows)", event.src_path, len(new_flows_df))

                self.buffer = pd.concat([self.buffer, new_flows_df], ignore_index=True)

                # Do inference in batches
                if len(self.buffer) >= MIN_INFERENCE_BATCH:
                    self.process_flows(self.buffer)
                    self.buffer = pd.DataFrame()
                else:
                    logging.info("Buffered %d flows (need %d to run inference)",
                                len(self.buffer), MIN_INFERENCE_BATCH)

def setup_logging():
    """
    Configure logging to output/log.txt and stdout.
    """
    log_dir = "output"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")
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

def find_lan_interface():
    """Return the first interface with a private/LAN IPv4 address, or 'eth0'."""
    import ipaddress
    for iface in netifaces.interfaces():
        addrs = netifaces.ifaddresses(iface).get(netifaces.AF_INET, [])
        for addr in addrs:
            if ipaddress.ip_address(addr['addr']).is_private and not addr['addr'].startswith('127.'):
                return iface
    return 'eth0'

def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iface', default=None, help='Network interface to capture (auto-detects LAN interface if omitted)')
    parser.add_argument("--model-path", required=True, help="Path to load the trained model from")
    parser.add_argument("--append-csv", action="store_true", help="Append to out.csv instead of overwriting on each garbage collect")
    parser.add_argument("--trace-path", default=None, help="Path to log post-processed features, predicted labels, and confidences")
    parser.add_argument("--data-path", default=None, help="Directory to store the SQLite database (default: ./data)")
    args = parser.parse_args()

    if args.iface is None:
        args.iface = find_lan_interface()
        logging.info(f"Auto-detected LAN interface: {args.iface}")

    # Tune pyflowmeter timeouts to produce shorter flows similar to CICFlowMeter.
    # Lower idle timeout so flows are finalized sooner.
    import pyflowmeter.flow_session as _fs
    _fs.EXPIRED_UPDATE = 5
    _fs.FlowSession.GARBAGE_COLLECT_PACKETS = 100

    if args.append_csv:
        # Monkey-patch pyflowmeter to append to CSV instead of overwriting.
        _original_init = _fs.FlowSession.__init__

        def _patched_init(self, *a, **kw):
            _original_init(self, *a, **kw)
            if self.to_csv and os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0:
                import csv
                output = open(self.output_file, "a", newline="")
                self.csv_writer = csv.writer(output)
                self.csv_line = 1  # skip writing header

        _fs.FlowSession.__init__ = _patched_init

    sniffer = None
    flows_csv = '/tmp/idps_flows.csv'

    sniffer = create_sniffer(
        input_interface=args.iface,
        to_csv=True,
        output_file=flows_csv
    )

    scaler_path = args.model_path + ".scaler.joblib"
    logging.info(f"Loading scaler from {scaler_path}")
    scaler = joblib.load(scaler_path)

    logging.info(f"Loading model weights from {args.model_path}")
    model = load_model_weights(args.model_path, num_features=scaler.n_features_in_)

    if args.data_path:
        set_data_path(args.data_path)
    # Ensure database and tables exist before starting
    get_connection().close()
    logging.info(f"Database ready at {get_db_path()}")

    observer = Observer()
    observer.schedule(FlowFileHandler(model, scaler, trace_path=args.trace_path), path='/tmp/', recursive=False)
    observer.start()
    sniffer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        sniffer.stop()

    observer.join()

if __name__ == "__main__":
    main()