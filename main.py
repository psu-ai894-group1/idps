import argparse
import logging
import pandas as pd
from flows_to_tensors import flows_to_tensors
from cic_to_flowmeter import cic_to_pyflowmeter_columns
from model import train

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Convert network flows CSV to GNN tensors.")
    parser.add_argument("--csv-path", required=True, help="Path to the flows CSV file")
    args = parser.parse_args()

    logging.info(f"Loading flows from {args.csv_path}")
    flows_df = pd.read_csv(args.csv_path)
    logging.info("Converting CIC-IDS-2017 columns to pyflowmeter columns")
    flows_df = cic_to_pyflowmeter_columns(flows_df)
    logging.info(f"Loaded {len(flows_df)} rows from {args.csv_path}")
    logging.info("Converting flows to tensors")
    node_features, adjacency, labels = flows_to_tensors(flows_df)
    logging.info("Flows converted to tensors")
    
    logging.info(f"Node features shape: {node_features.shape}")
    logging.info(f"Node features dtype: {node_features.dtype}")
    logging.info(f"Adjacency dense shape: {adjacency.dense_shape.numpy()}")
    logging.info(f"Adjacency non-zero entries: {adjacency.values.shape[0]}")
    logging.info(f"Labels: {labels}")

    logging.info("Training model")
    model = train(node_features, adjacency, labels)
    logging.info("Model trained")
    logging.info(model.summary())

if __name__ == "__main__":
    main()
