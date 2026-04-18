"""
AI-894
This file converts network flow data into graph tensors for use in Graph Neural Networks. It constructs nodes from flow features and edges based on communication patterns.
Authors: Karla Gonzalez Caballero (kxg5613@psu.edu), Christopher Umbel (czu5008@psu.edu)
"""
import numpy as np
import pandas as pd
import logging
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from config import class_names, feature_names, use_features
import tensorflow as tf

def flows_to_tensors(flows_df, scaler=None, log_transform=False):
    """
    Convert flows from pyflowmeter format to tensors for use in GNNs.

    This style of graph uses flows as nodes and edges based on behaviors as is described in:
    https://link.springer.com/article/10.1186/s42400-024-00296-8

    If scaler is None, a new StandardScaler is fitted on the data (training).
    If scaler is provided, it is used to transform the data without refitting (inference).
    If log_transform is True, apply log1p to numeric features before scaling.
    This is needed when using raw pyflowmeter output, since the training data
    (CIC-IDS-2017 cleaned) is already log-transformed.
    """
    flow_count = len(flows_df)
    logging.info(f"Flow count: {flow_count}")

    available_feature_names = [f for f in feature_names if f in flows_df.columns]
    if use_features != 'all':
        available_feature_names = [f for f in available_feature_names
                                   if f in use_features]
    logging.info(f"Available feature names: {available_feature_names}")

    node_features = flows_df[available_feature_names].values
    if log_transform:
        logging.info("Applying log1p transform to numeric features")
        node_features = np.log1p(np.abs(node_features)) * np.sign(node_features)
    if scaler is None:
        scaler = StandardScaler()
        logging.info("Fitting and scaling node features")
        node_features = scaler.fit_transform(node_features)
    else:
        logging.info("Scaling node features with pre-fitted scaler")
        node_features = scaler.transform(node_features)
    logging.info("Converting node features to TensorFlow constant")
    node_features = tf.constant(node_features, dtype=tf.float32)

    logging.info("Creating flow edges")
    adjacency = create_flow_edges_sparse(flows_df)
    logging.info("Flows converted to tensors")

    if "label" in flows_df.columns:
        name_to_idx = {name: i for i, name in enumerate(class_names)}
        label_series = flows_df["label"]
        if isinstance(label_series, pd.DataFrame):
            label_series = label_series.iloc[:, -1]
        label_strs = label_series.astype(str).str.strip().to_numpy(dtype=str)
        labels = np.array([name_to_idx[s] for s in label_strs], dtype=np.int32)
        labels = tf.constant(labels, dtype=tf.int32)
    else:
        labels = None

    return node_features, adjacency, labels, scaler
 
def create_flow_edges_sparse(flows_df, max_edges_per_group=50):
    """
    Create sparse adjacency matrix for flows in a memory efficient way.

    Flows are connected when they share the same (src_ip, dst_ip) pair
    (in either direction), meaning they belong to the same host-to-host
    communication channel. This avoids the near-fully-connected graph that
    results from connecting all flows sharing any single IP.

    Large groups are randomly sampled down to max_edges_per_group to keep
    the graph sparse.

    https://pytorch-geometric.readthedocs.io/en/2.5.1/advanced/sparse_tensor.html
    """
    n = len(flows_df)
    src_ip = flows_df["src_ip"].astype(str).to_numpy(dtype=str, na_value='nan')
    dst_ip = flows_df["dst_ip"].astype(str).to_numpy(dtype=str, na_value='nan')

    # Group flows by canonical (sorted) IP pair so A->B and B->A are the same group
    pair_to_flows = defaultdict(list)
    for i in range(n):
        s, d = str(src_ip[i]), str(dst_ip[i])
        if s == 'nan' or d == 'nan':
            continue
        pair = (min(s, d), max(s, d))
        pair_to_flows[pair].append(i)

    rows = []
    cols = []

    for flow_list in pair_to_flows.values():
        flow_list = sorted(set(flow_list))
        k = len(flow_list)

        if k < 2:
            continue

        # For large groups, sample edges to keep the graph sparse
        if k > max_edges_per_group:
            flow_list = list(np.random.choice(flow_list, max_edges_per_group, replace=False))
            flow_list.sort()
            k = len(flow_list)

        for idx in range(k):
            for jdx in range(idx + 1, k):
                rows.append(flow_list[idx])
                cols.append(flow_list[jdx])

    if not rows:
        return csr_matrix((np.ones(n), (range(n), range(n))), shape=(n, n))

    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    data = np.ones(len(rows), dtype=np.float32)

    upper = csr_matrix((data, (rows, cols)), shape=(n, n))
    adj = upper + upper.T

    return adj

