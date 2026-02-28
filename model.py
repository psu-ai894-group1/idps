import tensorflow as tf
import numpy as np
import logging
import config
from scipy.sparse import diags


class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[-1], self.output_dim),
            initializer="glorot_uniform",
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.output_dim,),
            initializer="zeros",
        )

    def call(self, x, adj_norm):
        xw = tf.matmul(x, self.kernel)
        # This would overflow GPU limits. We had to move it to CPU in order to work as per
        # this this issue thread on GitHub: https://github.com/tkipf/keras-gcn/issues/51
        with tf.device("/CPU:0"):
            out = tf.sparse.sparse_dense_matmul(adj_norm, xw)
        return self.activation(out + self.bias)

class GCNClassifier(tf.keras.Model):
    def __init__(self, hidden_dim=config.hidden_dim, num_classes=2):
        super().__init__()
        self.gcn1 = GCNLayer(hidden_dim, activation="relu")
        self.gcn2 = GCNLayer(hidden_dim, activation="relu")
        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, x, adj_norm):
        h = self.gcn1(x, adj_norm)
        h = self.gcn2(h, adj_norm)
        return self.classifier(h)

def _normalize_adjacency(adj):
    """
    Compute D^{-1/2} (A + I) D^{-1/2} using scipy sparse matrices.
    
    https://github.com/tkipf/gcn/blob/master/gcn/utils.py
    https://tkipf.github.io/graph-convolutional-networks/
    https://github.com/tkipf/pygcn/issues/47
    """
    n = adj.shape[0]
    adj_hat = adj + diags(np.ones(n))
    degree = np.array(adj_hat.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(degree > 0, np.power(degree, -0.5), 0.0)
    D_inv_sqrt = diags(d_inv_sqrt)
    return (D_inv_sqrt @ adj_hat @ D_inv_sqrt).tocsr()


def _scipy_to_tf_sparse(csr_mat):
    """Convert a scipy sparse matrix to a tf.SparseTensor on CPU."""
    coo = csr_mat.tocoo()
    indices = np.column_stack((coo.row, coo.col)).astype(np.int64)
    values = coo.data.astype(np.float32)
    with tf.device("/CPU:0"):
        sparse = tf.SparseTensor(
            indices=indices, values=values,
            dense_shape=csr_mat.shape
        )
        return tf.sparse.reorder(sparse)


def _sample_subgraph(batch_nodes, adj_csr, node_features, max_neighbors, diag, num_hops=2):
    """
    Sample a k-hop subgraph around batch_nodes with neighbor sampling.

    Only sampled edges (plus self-loops) are included in the subgraph adjacency,
    not all edges between subgraph nodes. This keeps the subgraph sparse even when
    the full graph is dense. Edge values come from the pre-normalized full adjacency
    so normalization coefficients stay correct.

    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
    https://pytorch-geometric.readthedocs.io/en/2.5.2/tutorial/neighbor_loader.html
    """
    all_nodes = set(batch_nodes.tolist())
    frontier = batch_nodes.tolist()
    edges = {}  # (global_src, global_dst) -> pre-normalized value

    for _ in range(num_hops):
        next_frontier = []
        for node in frontier:
            start, end = adj_csr.indptr[node], adj_csr.indptr[node + 1]
            neighbors = adj_csr.indices[start:end]
            values = adj_csr.data[start:end]
            if len(neighbors) > max_neighbors:
                chosen = np.random.choice(len(neighbors), max_neighbors, replace=False)
                neighbors = neighbors[chosen]
                values = values[chosen]
            for nb, val in zip(neighbors, values):
                nb_int = int(nb)
                val_f = float(val)
                edges[(node, nb_int)] = val_f
                edges[(nb_int, node)] = val_f
                if nb_int not in all_nodes:
                    all_nodes.add(nb_int)
                    next_frontier.append(nb_int)
        frontier = next_frontier

    sub_nodes = np.array(sorted(all_nodes))
    n_sub = len(sub_nodes)

    # Map global indices to local
    global_to_local = np.full(adj_csr.shape[0], -1, dtype=np.int64)
    global_to_local[sub_nodes] = np.arange(n_sub)
    batch_local = global_to_local[batch_nodes]

    # Add self-loops from the pre-normalized adjacency diagonal
    for node in sub_nodes:
        n = int(node)
        edges[(n, n)] = float(diag[n])

    # Build tf.SparseTensor from sampled edges only
    src, dst = zip(*edges.keys())
    vals = np.array(list(edges.values()), dtype=np.float32)
    l_src = global_to_local[np.array(src)]
    l_dst = global_to_local[np.array(dst)]
    indices = np.column_stack((l_src, l_dst)).astype(np.int64)

    with tf.device("/CPU:0"):
        sub_adj = tf.SparseTensor(indices=indices, values=vals, dense_shape=[n_sub, n_sub])
        sub_adj = tf.sparse.reorder(sub_adj)

    sub_features = tf.gather(node_features, sub_nodes)
    return sub_features, sub_adj, batch_local


def train(node_features, adjacency, labels,
          hidden_dim=config.hidden_dim, num_classes=None, epochs=config.epochs,
          batch_size=config.batch_size, max_neighbors=config.max_neighbors, lr=config.learning_rate,
          weight_decay=config.weight_decay):
    labels = tf.cast(tf.constant(labels), tf.int32)
    if num_classes is None:
        num_classes = int(tf.reduce_max(labels).numpy()) + 1

    logging.info("Normalizing adjacency matrix")
    adj_norm = _normalize_adjacency(adjacency)
    diag = adj_norm.diagonal()

    n_nodes = node_features.shape[0]

    model = GCNClassifier(hidden_dim=hidden_dim, num_classes=num_classes)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(n_nodes)
        epoch_loss = 0.0
        epoch_correct = 0
        n_batches = 0

        for start in range(0, n_nodes, batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_labels = tf.gather(labels, batch_idx)

            sub_features, sub_adj, batch_local = _sample_subgraph(
                batch_idx, adj_norm, node_features, max_neighbors, diag)

            with tf.GradientTape() as tape:
                logits = model(sub_features, sub_adj)
                batch_logits = tf.gather(logits, batch_local)
                loss = loss_fn(batch_labels, batch_logits)

            # Gradients computation is moved to CPU to avoid OOM.
            with tf.device("/CPU:0"):
                grads = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            preds = tf.argmax(batch_logits, axis=1, output_type=tf.int32)
            epoch_correct += int(tf.reduce_sum(
                tf.cast(tf.equal(preds, batch_labels), tf.int32)
            ).numpy())
            epoch_loss += loss.numpy()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        accuracy = epoch_correct / n_nodes
        logging.info(
            f"Epoch {epoch:>3d}/{epochs}  loss={avg_loss:.4f}  "
            f"accuracy={accuracy:.4f}"
        )

    return model


def predict(model, node_features, adjacency):
    """
    Run inference on a graph and return predicted class indices.

    Full-graph forward pass is used for inference since there are
    fewer memory concerns without worrying about gradients.
    """
    adj_norm = _normalize_adjacency(adjacency)
    adj_tf = _scipy_to_tf_sparse(adj_norm)
    logits = model(node_features, adj_tf)
    return tf.argmax(logits, axis=1, output_type=tf.int32).numpy()
