import argparse
import logging
import os
import dotenv
dotenv.load_dotenv()

from pyflowmeter.sniffer import create_sniffer
from pyflowmeter.features.flow_bytes import FlowBytes

# Monkey-patch for pyflowmeter bug: ZeroDivisionError in get_bulk_rate
# when a flow has no backward packets.
_original_get_bulk_rate = FlowBytes.get_bulk_rate

def _patched_get_bulk_rate(self, direction):
    try:
        return _original_get_bulk_rate(self, direction)
    except ZeroDivisionError:
        return 0

FlowBytes.get_bulk_rate = _patched_get_bulk_rate
import time
import joblib
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from flows_to_tensors import flows_to_tensors
from config import class_names

import tensorflow as tf
from model import predict, load_model_weights

MIN_INFERENCE_BATCH = 50

class FlowFileHandler(FileSystemEventHandler):
    def __init__(self, model, scaler, trace_path=None):
        self.model = model
        self.scaler = scaler
        self.pos = 0
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

        if self.trace_path:
            self._write_trace(node_features, preds, confidences)

    def _write_trace(self, node_features, preds, confidences):
        from config import feature_names, use_features
        available = [f for f in feature_names if f in use_features]
        features_np = node_features.numpy() if hasattr(node_features, 'numpy') else node_features
        trace_df = pd.DataFrame(features_np, columns=available)
        trace_df['predicted_label'] = [class_names[p] for p in preds]
        trace_df['confidence'] = confidences

        write_header = not self.trace_header_written and not (
            os.path.exists(self.trace_path) and os.path.getsize(self.trace_path) > 0
        )
        trace_df.to_csv(self.trace_path, mode='a', index=False, header=write_header)
        self.trace_header_written = True

    def on_modified(self, event):
        if event.src_path.endswith('.csv') and os.path.exists(event.src_path) and os.path.getsize(event.src_path) > 0:
            logging.info("New flows detected: %s", event.src_path)
            new_flows_df = pd.read_csv(event.src_path)
            new_flows_df = new_flows_df.iloc[self.pos:].copy()
            self.pos += len(new_flows_df)
            logging.info("Flow count: %d", len(new_flows_df))

            if len(new_flows_df) > 0:
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

def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iface', default='eth0', help='Network interface to capture (default: eth0)')
    parser.add_argument("--model-path", required=True, help="Path to load the trained model from")
    parser.add_argument("--append-csv", action="store_true", help="Append to out.csv instead of overwriting on each garbage collect")
    parser.add_argument("--trace-path", default=None, help="Path to log post-processed features, predicted labels, and confidences")
    args = parser.parse_args()

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
    output_dir = './output/'

    sniffer = create_sniffer(
        input_interface=args.iface,
        to_csv=True,
        output_file=os.path.join(output_dir, 'out.csv')
    )

    scaler_path = args.model_path + ".scaler.joblib"
    logging.info(f"Loading scaler from {scaler_path}")
    scaler = joblib.load(scaler_path)

    logging.info(f"Loading model weights from {args.model_path}")
    model = load_model_weights(args.model_path, num_features=scaler.n_features_in_)

    observer = Observer()
    observer.schedule(FlowFileHandler(model, scaler, trace_path=args.trace_path), path=output_dir, recursive=False)
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