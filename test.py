import argparse
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from flows_to_tensors import flows_to_tensors
from model import predict, GCNClassifier, GCNLayer
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

def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained GCN model.")
    parser.add_argument("--model-path", required=True, help="Path to load the saved model from")
    args = parser.parse_args()

    logging.info(f"Loading model from {args.model_path}")
    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={"GCNClassifier": GCNClassifier, "GCNLayer": GCNLayer},
    )

    mock_flow = pd.DataFrame([{
        "src_ip": "192.168.1.10",
        "dst_ip": "10.0.0.1",
        "flow_duration": 12345,
        "tot_fwd_pkts": 6,
        "tot_bwd_pkts": 4,
        "totlen_fwd_pkts": 720,
        "totlen_bwd_pkts": 1480,
        "fwd_pkt_len_mean": 120.0,
        "bwd_pkt_len_mean": 370.0,
        "flow_byts_s": 178.2,
        "flow_pkts_s": 0.81,
        "fwd_iat_mean": 2500.0,
        "bwd_iat_mean": 3100.0,
        "fin_flag_cnt": 1,
        "syn_flag_cnt": 1,
        "rst_flag_cnt": 0,
        "psh_flag_cnt": 1,
        "ack_flag_cnt": 1,
        "urg_flag_cnt": 0,
        "cwe_flag_count": 0,
        "ece_flag_cnt": 0,
        "down_up_ratio": 0.5,
        "pkt_size_avg": 220.0,
        "label": "BENIGN",
    }])

    logging.info("Converting mock flow to tensors")
    node_features, adjacency, labels = flows_to_tensors(mock_flow)

    logging.info("Running inference on mock flow")
    preds = predict(model, node_features, adjacency)

    predicted_class = class_names[preds[0]]
    true_class = class_names[labels.numpy()[0]]
    logging.info(f"True label: {true_class}")
    logging.info(f"Predicted:  {predicted_class}")


if __name__ == "__main__":
    main()
