import argparse
import logging
import os
import dotenv
dotenv.load_dotenv()

from pyflowmeter.sniffer import create_sniffer
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
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.pos = 0
        self.buffer = pd.DataFrame()
        logging.info("FlowFileHandler initialized")

    def process_flows(self, flows_df):
        logging.info("Processing flows: %d", len(flows_df))
        node_features, adjacency, _, _ = flows_to_tensors(flows_df, scaler=self.scaler, log_transform=True)
        preds, confidences = predict(self.model, node_features, adjacency)

        for i in range(len(flows_df)):
            label = class_names[preds[i]]
            logging.info("Row %d predicted label: %s (confidence: %.4f)", i, label, confidences[i])

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
    args = parser.parse_args()

    # Tune pyflowmeter timeouts to produce shorter flows similar to CICFlowMeter.
    # Lower idle timeout so flows are finalized sooner.
    import pyflowmeter.flow_session as _fs
    _fs.EXPIRED_UPDATE = 5
    _fs.FlowSession.GARBAGE_COLLECT_PACKETS = 100

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
    observer.schedule(FlowFileHandler(model, scaler), path=output_dir, recursive=False)
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