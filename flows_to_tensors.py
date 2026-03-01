import numpy as np
import logging
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from config import class_names, feature_names
import tensorflow as tf

def flows_to_tensors(flows_df):
    """
    Convert flows from pyflowmeter format to tensors for use in GNNs.

    This style of graph uses flows as nodes and edges based on behaviors as is described in:
    https://link.springer.com/article/10.1186/s42400-024-00296-8
    """
    flow_count = len(flows_df)
    logging.info(f"Flow count: {flow_count}")

    available_feature_names = [f for f in feature_names if f in flows_df.columns]
    logging.info(f"Available feature names: {available_feature_names}")

    node_features = flows_df[available_feature_names].values
    scaler = StandardScaler()
    logging.info("Scaling node features")
    node_features = scaler.fit_transform(node_features)
    logging.info("Converting node features to TensorFlow constant")
    node_features = tf.constant(node_features, dtype=tf.float32)

    logging.info("Creating flow edges")
    adjacency = create_flow_edges_sparse(flows_df)
    logging.info("Flows converted to tensors")

    name_to_idx = {name: i for i, name in enumerate(class_names)}
    label_strs = flows_df["label"].values.astype(str)
    labels = np.array([name_to_idx[s.strip()] for s in label_strs], dtype=np.int32)
    labels = tf.constant(labels, dtype=tf.int32)

    return node_features, adjacency, labels
 
def create_flow_edges_sparse(flows_df):
    """
    Create sparse adjacency matrix for flows in a memory efficient way.

    https://pytorch-geometric.readthedocs.io/en/2.5.1/advanced/sparse_tensor.html
    """
    n = len(flows_df)
    src_ip = flows_df["src_ip"].values
    dst_ip = flows_df["dst_ip"].values

    ip_to_flows = defaultdict(list)

    for i in range(n):
        ip_to_flows[src_ip[i]].append(i)
        ip_to_flows[dst_ip[i]].append(i)

    rows = []
    cols = []

    for flow_list in ip_to_flows.values():
        flow_list = sorted(set(flow_list))
        k = len(flow_list)

        if k < 2:
            continue

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

