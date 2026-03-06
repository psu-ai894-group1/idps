import argparse
import logging
import os
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from flows_to_tensors import flows_to_tensors
from cic_to_flowmeter import cic_to_pyflowmeter_columns
from model import predict, load_model_weights
from config import class_names

logging.basicConfig(level=logging.INFO)

gpus = tf.config.experimental.list_physical_devices('GPU')
logging.info(gpus)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_visible_devices(gpu, 'GPU')
            tf.config.experimental.set_memory_growth(gpu, True)

        logical_gpus = tf.config.list_logical_devices('GPU')
    except Exception as e:
        logging.error("Exception setting up GPUs")
        logging.error(e)

def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    logging.info(f"Test accuracy: {accuracy:.4f}")

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    logging.info(f"Classification report:\n{report}")

    cm = confusion_matrix(y_true, y_pred)
    header = "".ljust(20) + "".join(name.ljust(15) for name in class_names)
    logging.info(f"Confusion matrix (rows=true, cols=predicted):\n{header}")
    for i, row in enumerate(cm):
        row_str = class_names[i].ljust(20) + "".join(str(v).ljust(15) for v in row)
        logging.info(row_str)

    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Run batch inference with a trained GCN model.")
    parser.add_argument("--csv-path", required=True, help="Path to the flows CSV file")
    parser.add_argument("--model-path", required=True, help="Path to load the saved model from")
    args = parser.parse_args()

    scaler_path = args.model_path + ".scaler.joblib"
    logging.info(f"Loading scaler from {scaler_path}")
    scaler = joblib.load(scaler_path)

    logging.info(f"Loading flows from {args.csv_path}")
    flows_df = pd.read_csv(args.csv_path)
    flows_df = cic_to_pyflowmeter_columns(flows_df)
    logging.info(f"Loaded {len(flows_df)} rows from {args.csv_path}")

    logging.info("Converting flows to tensors")
    node_features, adjacency, labels, _ = flows_to_tensors(flows_df, scaler=scaler)

    logging.info(f"Loading model weights from {args.model_path}")
    model = load_model_weights(args.model_path, num_features=node_features.shape[1])

    logging.info("Running inference")
    preds, _ = predict(model, node_features, adjacency)
    true = labels.numpy()

    evaluate(true, preds)


if __name__ == "__main__":
    main()
