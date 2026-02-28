import argparse
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from flows_to_tensors import flows_to_tensors
from cic_to_flowmeter import cic_to_pyflowmeter_columns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from model import train, predict
from meta import class_names
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

gpus = tf.config.experimental.list_physical_devices('GPU')
logging.info(gpus)

if gpus:
  try:    
    for gpu in gpus:
      tf.config.experimental.set_visible_devices(gpu, 'GPU')
      tf.config.experimental.set_memory_growth(gpu, True)

    logical_gpus = tf.config.list_logical_devices('GPU')
    #logging.info(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except Exception as e:
    logging.error("EXCEPTION")
    logging.error(e)

def show_report(y_true, y_pred):
    logging.info(f"Test accuracy: {accuracy_score(y_true, y_pred):.4f}")

    present_labels = sorted(set(y_true) | set(y_pred))
    present_names = [class_names[i] for i in present_labels]

    print("\n" + classification_report(
        y_true, y_pred, labels=present_labels, target_names=present_names, zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    header = "  ".join(f"{n[:8]:>8}" for n in present_names)
    print(f"Confusion matrix (rows=true, cols=predicted):\n{'':>12}{header}")
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>8d}" for v in row)
        print(f"{present_names[i]:>12}{row_str}")

def main():
    parser = argparse.ArgumentParser(description="Convert network flows CSV to GNN tensors.")
    parser.add_argument("--csv-path", required=True, help="Path to the flows CSV file")
    args = parser.parse_args()

    logging.info(f"Loading flows from {args.csv_path}")
    flows_df = pd.read_csv(args.csv_path)
    logging.info("Converting CIC-IDS-2017 columns to pyflowmeter columns")
    flows_df = cic_to_pyflowmeter_columns(flows_df)
    logging.info(f"Loaded {len(flows_df)} rows from {args.csv_path}")

    train_df, test_df = train_test_split(flows_df, test_size=0.2, shuffle=True, random_state=42)
    logging.info(f"Train/test split: {len(train_df)} / {len(test_df)} rows")

    logging.info("Converting flows to tensors")
    node_features, adjacency, labels = flows_to_tensors(train_df)
    logging.info("Flows converted to tensors")
    
    logging.info(f"Node features shape: {node_features.shape}")
    logging.info(f"Node features dtype: {node_features.dtype}")
    logging.info(f"Adjacency dense shape: {adjacency.dense_shape.numpy()}")
    logging.info(f"Adjacency non-zero entries: {adjacency.values.shape[0]}")
    logging.info(f"Labels: {labels}")

    logging.info("Training model")
    model = train(node_features, adjacency, labels)
    logging.info("Model trained")

    # Inference on test set
    logging.info("Converting test set to tensors")
    test_features, test_adj, test_labels = flows_to_tensors(test_df)
    logging.info("Running inference on test set")
    preds = predict(model, test_features, test_adj)
    true = test_labels.numpy()

    show_report(true, preds)

if __name__ == "__main__":
    main()
