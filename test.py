"""
AI-894
Replicates the dashboard's "Import Test Data" flow from the command line:
loads a CIC-IDS-2017 CSV, runs GCN inference identically to dashboard.py,
persists results to the database, and prints a confusion matrix and metrics.
Authors: Karla Gonzalez Caballero (kxg5613@psu.edu), Christopher Umbel (czu5008@psu.edu)
"""
import argparse
import logging
import os

import joblib
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from cic_to_flowmeter import cic_to_pyflowmeter_columns
from config import class_names
from db import save_flows_and_inferences
from flows_to_tensors import flows_to_tensors
from model import (
    _normalize_adjacency,
    _scipy_to_tf_sparse,
    load_model_weights,
    predict,
)

logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "models", "idps.weights.h5")

gpus = tf.config.experimental.list_physical_devices('GPU')
logging.info(gpus)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_visible_devices(gpu, 'GPU')
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.list_logical_devices('GPU')
    except Exception as e:
        logging.error("Exception setting up GPUs")
        logging.error(e)

def print_metrics(true, preds, probs):
    acc = accuracy_score(true, preds)
    bal = balanced_accuracy_score(true, preds)
    f1  = f1_score(true, preds, average="macro", zero_division=0)
    ll  = log_loss(true, probs, labels=list(range(len(class_names))))

    logging.info(f"Accuracy:          {acc:.4f}")
    logging.info(f"Balanced accuracy: {bal:.4f}")
    logging.info(f"F1 (macro):        {f1:.4f}")
    logging.info(f"Log loss:          {ll:.4f}")

    report = classification_report(true, preds, target_names=class_names, zero_division=0)
    logging.info(f"Classification report:\n{report}")

    cm = confusion_matrix(true, preds, labels=list(range(len(class_names))))
    header = "".ljust(20) + "".join(name.ljust(15) for name in class_names)
    logging.info(f"Confusion matrix (rows=true, cols=predicted):\n{header}")
    for i, row in enumerate(cm):
        row_str = class_names[i].ljust(20) + "".join(str(v).ljust(15) for v in row)
        logging.info(row_str)

def main():
    parser = argparse.ArgumentParser(
        description="Run inference like the dashboard's Import Test Data flow."
    )
    parser.add_argument("--csv-path", required=True, help="Path to the flows CSV file")
    parser.add_argument("--model-path", default=DEFAULT_MODEL,
                        help="Path to load the saved model from")
    parser.add_argument("--no-save-db", action="store_true",
                        help="Skip writing flows and inferences to the database")
    args = parser.parse_args()

    scaler_path = args.model_path + ".scaler.joblib"
    logging.info(f"Loading scaler from {scaler_path}")
    scaler = joblib.load(scaler_path)

    logging.info(f"Loading flows from {args.csv_path}")
    flows_df = pd.read_csv(args.csv_path)
    flows_df = cic_to_pyflowmeter_columns(flows_df)
    logging.info(f"Loaded {len(flows_df)} rows from {args.csv_path}")

    logging.info("Converting flows to tensors")
    node_features, adjacency, labels, _ = flows_to_tensors(
        flows_df, scaler=scaler, log_transform=True
    )

    logging.info(f"Loading model weights from {args.model_path}")
    model = load_model_weights(args.model_path, num_features=node_features.shape[1])

    logging.info("Running inference")
    preds, confidences = predict(model, node_features, adjacency)

    adj_tf = _scipy_to_tf_sparse(_normalize_adjacency(adjacency))
    probs  = tf.nn.softmax(model(node_features, adj_tf), axis=1).numpy()

    if not args.no_save_db:
        logging.info("Saving flows and inferences to database")
        save_flows_and_inferences(flows_df, preds, confidences, class_names)

    if labels is None:
        logging.warning("CSV has no labels; skipping metrics.")
        return

    print_metrics(labels.numpy(), preds, probs)

if __name__ == "__main__":
    main()
