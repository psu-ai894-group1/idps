import argparse
import logging
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from flows_to_tensors import flows_to_tensors
from cic_to_flowmeter import cic_to_pyflowmeter_columns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from model import train, predict, GCNClassifier, GCNLayer
import config
from config import class_names
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

log_dir = "output"
os.makedirs(log_dir, exist_ok=True)
train_log_path = os.path.join(log_dir, "train.log")
file_handler = logging.FileHandler(train_log_path, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.getLogger().addHandler(file_handler)

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
    # TODO: Add more metrics i.e. precision, recall, f1-score, etc.
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
    parser = argparse.ArgumentParser(description="Convert network flows CSV to GNN tensors.")
    parser.add_argument("--csv-path", required=True, help="Path to the flows CSV file")
    parser.add_argument("--model-path", required=True, help="Path to save the trained model weights")
    args = parser.parse_args()

    logging.info(f"Loading flows from {args.csv_path}")
    flows_df = pd.read_csv(args.csv_path)
    logging.info("Converting CIC-IDS-2017 columns to pyflowmeter columns")
    flows_df = cic_to_pyflowmeter_columns(flows_df)
    logging.info(f"Loaded {len(flows_df)} rows from {args.csv_path}")

    train_df, test_df = train_test_split(flows_df, test_size=config.test_size, shuffle=True, random_state=42)
    # Split remaining data into train and cross-validation sets.
    xval_fraction = config.xval_size / (1 - config.test_size)
    train_df, xval_df = train_test_split(train_df, test_size=xval_fraction, shuffle=True, random_state=42)
    logging.info(f"Train/xval/test split: {len(train_df)} / {len(xval_df)} / {len(test_df)} rows")

    logging.info("Converting training flows to tensors")
    node_features, adjacency, labels, scaler = flows_to_tensors(train_df)
    logging.info("Flows converted to tensors")

    scaler_path = args.model_path + ".scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logging.info(f"Saved scaler to {scaler_path}")

    logging.info(f"Node features shape: {node_features.shape}")
    logging.info(f"Node features dtype: {node_features.dtype}")
    logging.info(f"Adjacency shape: {adjacency.shape}")
    logging.info(f"Adjacency non-zero entries: {adjacency.nnz}")
    logging.info(f"Labels: {labels}")

    logging.info("Converting cross-validation set to tensors")
    xval_features, xval_adj, xval_labels, _ = flows_to_tensors(xval_df, scaler=scaler)

    logging.info("Training model")
    model = train(node_features, adjacency, labels,
                  xval_features=xval_features, xval_adj=xval_adj, xval_labels=xval_labels)
    logging.info("Model trained")

    # Inference on test set — reuse the training scaler
    logging.info("Converting test set to tensors")
    test_features, test_adj, test_labels, _ = flows_to_tensors(test_df, scaler=scaler)
    logging.info("Running inference on test set")
    preds, _ = predict(model, test_features, test_adj)
    true = test_labels.numpy()
    accuracy = evaluate(true, preds)
    logging.info(f"Test accuracy: {accuracy:.4f}")

    logging.info("Loading best checkpoint weights for final save")
    model.load_weights("output/checkpoint.weights.h5")
    model.save_weights(args.model_path)
    logging.info(f"Saved model weights to {args.model_path}")

if __name__ == "__main__":
    main()
