import tensorflow as tf
import numpy as np
import logging
import config
from scipy.sparse import diags

class GCNLayer(tf.keras.layers.Layer):
    """
    GCN hidden layer. Since all hidden layers are the same, we can use a single layer class.
    """

    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[-1], self.output_dim),
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
        return tf.nn.relu(out + self.bias)

class GCNClassifier(tf.keras.Model):
    """
    A GCN classifier model. Configurable hidden GCN+Dropout layers, multi-class classification output layer.

    https://tkipf.github.io/graph-convolutional-networks/
    """

    def __init__(self, hidden_dim=config.hidden_dim, num_classes=2,
                 dropout_rate=config.dropout_rate,
                 num_hidden_layers=config.num_hidden_layers, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.num_hidden_layers = num_hidden_layers
        self.gcn_layers = []
        self.dropout_layers = []
        for _ in range(num_hidden_layers):
            self.gcn_layers.append(GCNLayer(hidden_dim))
            self.dropout_layers.append(tf.keras.layers.Dropout(dropout_rate))
        self.classifier = tf.keras.layers.Dense(num_classes)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
            "num_hidden_layers": self.num_hidden_layers,
        })
        return config

    def call(self, x, adj_norm, training=False):
        h = x
        for gcn, dropout in zip(self.gcn_layers, self.dropout_layers):
            h = gcn(h, adj_norm)
            h = dropout(h, training=training)
        return self.classifier(h)

def load_model_weights(weights_path, num_features, num_classes=len(config.class_names)):
    """
    Create a GCNClassifier, build it with a dummy input, and load saved weights.
    """
    model = GCNClassifier(num_classes=num_classes)
    dummy_x = tf.zeros((1, num_features))
    dummy_adj = tf.SparseTensor(indices=[[0, 0]], values=[1.0], dense_shape=[1, 1])
    model(dummy_x, dummy_adj)
    model.load_weights(weights_path)
    return model


def _normalize_adjacency(adj):
    """
    Compute D^{-1/2} (A + I) D^{-1/2} using scipy sparse matrices.
    
    This prevents nodes with high degree (many connections) from dominating the learning process.

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
    """
    Convert a scipy sparse matrix to a tf.SparseTensor on CPU.
    """
    # Convert from CSR to COO format to get row and column indices as that's what
    # TensorFlow needs
    coo = csr_mat.tocoo()
    indices = np.column_stack((coo.row, coo.col)).astype(np.int64)
    values = coo.data.astype(np.float32)

    # Convert to tf.SparseTensor on CPU to avoid OOM.
    with tf.device("/CPU:0"):
        sparse = tf.SparseTensor(
            indices=indices, values=values,
            dense_shape=csr_mat.shape
        )
        return tf.sparse.reorder(sparse)


def _sample_subgraph(batch_nodes, adj_csr, node_features, max_neighbors, diag, num_hops=2):
    """
    Sample a subgraph around batch_nodes with neighbor sampling within k hops.

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

    # Breadth-first search loop, stopping at num_hops distance.
    for _ in range(num_hops):
        next_frontier = []
        
        # Sample neighbors for each node in the frontier.
        for node in frontier:
            start, end = adj_csr.indptr[node], adj_csr.indptr[node + 1]
            neighbors = adj_csr.indices[start:end]
            values = adj_csr.data[start:end]

            # To conserve memory, we sample a subset of the neighbors.
            if len(neighbors) > max_neighbors:
                chosen = np.random.choice(len(neighbors), max_neighbors, replace=False)
                neighbors = neighbors[chosen]
                values = values[chosen]

            # Add edges to the subgraph.
            for nb, val in zip(neighbors, values):
                nb_int = int(nb)
                val_f = float(val)
                edges[(node, nb_int)] = val_f
                edges[(nb_int, node)] = val_f
                
                # Add new nodes to the frontier if they are not already in the subgraph.
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
          hidden_dim=config.hidden_dim, num_classes=len(config.class_names), epochs=config.epochs,
          batch_size=config.batch_size, max_neighbors=config.max_neighbors, lr=config.learning_rate,
          weight_decay=config.weight_decay, clipnorm=config.clipnorm,
          class_reweighting=config.class_reweighting,
          xval_features=None, xval_adj=None, xval_labels=None):
    """
    Train a GCN classifier on the given node features and adjacency matrix.

    If xval_features/xval_adj/xval_labels are provided, cross-validation
    accuracy is computed at the end of each epoch.

    Returns the trained model.
    """

    labels = tf.cast(tf.constant(labels), tf.int32)

    logging.info("Normalizing adjacency matrix")
    adj_norm = _normalize_adjacency(adjacency)
    diag = adj_norm.diagonal()

    n_nodes = node_features.shape[0]

    labels_np = labels.numpy()
    if class_reweighting:
        # Use weighted loss to deal with class imbalance.
        # Compute class weights inversely proportional to class frequency.
        # https://gist.github.com/jonnyli1125/5384bb9a41caaac983f1cd737359c6c2
        # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
        # Use smooth weights to penalize more for getting BENIGN wrong
        # https://medium.com/gumgum-tech/handling-class-imbalance-by-introducing-sample-weighting-in-the-loss-function-3bdebd8203b4
        class_counts = np.bincount(labels_np, minlength=num_classes).astype(np.float32)
        raw_weights = np.where(class_counts > 0, n_nodes / (num_classes * class_counts), 0.0)
        class_weights = np.sqrt(raw_weights)
        sample_weights = tf.constant(class_weights[labels_np], dtype=tf.float32)
        logging.info(f"Class weights: {dict(enumerate(class_weights.tolist()))}")
    else:
        sample_weights = tf.ones(n_nodes, dtype=tf.float32)
        logging.info("Class reweighting disabled")

    model = GCNClassifier(hidden_dim=hidden_dim, num_classes=num_classes)
    # AdamW optimizer is used for training to avoid overfitting.
    # Gradient clipping is used to prevent exploding gradients.
    # https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay, clipnorm=clipnorm)
    # Sparse categorical cross-entropy loss with no built-in reduction so we can apply class weights.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    # Early stopping: stop if loss doesn't improve for `patience` epochs.
    best_loss = float('inf')
    patience = config.early_stopping_patience
    epochs_without_improvement = 0

    # Pre-compute cross-validation adjacency normalization
    if xval_features is not None:
        xval_adj_norm = _normalize_adjacency(xval_adj)
        xval_adj_tf = _scipy_to_tf_sparse(xval_adj_norm)

    # Epoch loop
    for epoch in range(1, epochs + 1):
        # Shuffle the node indices for each epoch.
        indices = np.random.permutation(n_nodes)
        epoch_loss = 0.0
        epoch_correct = 0
        n_batches = 0

        # Mini-batch loop
        for start in range(0, n_nodes, batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_labels = tf.gather(labels, batch_idx)

            sub_features, sub_adj, batch_local = _sample_subgraph(
                batch_idx, adj_norm, node_features, max_neighbors, diag)

            with tf.GradientTape() as tape:
                logits = model(sub_features, sub_adj, training=True)
                batch_logits = tf.gather(logits, batch_local)
                # Compute class weights for this batch to deal with class imbalance 
                batch_weights = tf.gather(sample_weights, batch_idx)
                per_sample_loss = loss_fn(batch_labels, batch_logits)
                # Use weighted loss to deal with class imbalance.
                loss = tf.reduce_mean(per_sample_loss * batch_weights)

            # Gradients computation is moved to CPU to avoid OOM.
            with tf.device("/CPU:0"):
                # Backpropagation!
                grads = tape.gradient(loss, model.trainable_variables)

            # Adjust the model's weights based on the computed gradients (train)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Compute training accuracy for this batch.
            preds = tf.argmax(batch_logits, axis=1, output_type=tf.int32)
            epoch_correct += int(tf.reduce_sum(
                tf.cast(tf.equal(preds, batch_labels), tf.int32)
            ).numpy())
            epoch_loss += loss.numpy()
            n_batches += 1

        # Compute average loss and accuracy for this epoch.
        avg_loss = epoch_loss / n_batches
        accuracy = epoch_correct / n_nodes

        # Compute cross-validation loss and accuracy if available.
        if xval_features is not None:
            xval_logits = model(xval_features, xval_adj_tf, training=False)
            xval_per_sample_loss = loss_fn(xval_labels, xval_logits)
            xval_loss = float(tf.reduce_mean(xval_per_sample_loss).numpy())
            xval_preds = tf.argmax(xval_logits, axis=1, output_type=tf.int32).numpy()
            xval_correct = int(np.sum(xval_preds == xval_labels.numpy()))
            xval_acc = xval_correct / len(xval_labels)
            logging.info(
                f"Epoch {epoch:>3d}/{epochs}  loss={avg_loss:.4f}  "
                f"train_acc={accuracy:.4f}  xval_loss={xval_loss:.4f}  "
                f"xval_acc={xval_acc:.4f}"
            )
        else:
            xval_loss = None
            logging.info(
                f"Epoch {epoch:>3d}/{epochs}  loss={avg_loss:.4f}  "
                f"accuracy={accuracy:.4f}"
            )

        # Early stopping check. Use validation loss when available, otherwise training loss.
        monitored_loss = xval_loss if xval_loss is not None else avg_loss
        if monitored_loss < best_loss:
            best_loss = monitored_loss
            epochs_without_improvement = 0
            model.save_weights("output/checkpoint.weights.h5")
            logging.info(f"Checkpoint saved (loss={monitored_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logging.info(f"Early stopping at epoch {epoch} (no improvement in "
                             f"{'xval' if xval_loss is not None else 'training'} loss "
                             f"for {patience} epochs)")
                break

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
    probs = tf.nn.softmax(logits, axis=1)
    threshold = getattr(config, 'attack_confidence_threshold', 0.0)
    if threshold > 0.0:
        # Zero out attack classes (index > 0) that are below threshold
        attack_probs = probs[:, 1:]
        attack_mask = tf.cast(attack_probs >= threshold, tf.float32)
        masked_attack = attack_probs * attack_mask
        masked_probs = tf.concat([probs[:, :1], masked_attack], axis=1)
        preds = tf.argmax(masked_probs, axis=1, output_type=tf.int32).numpy()
    else:
        preds = tf.argmax(probs, axis=1, output_type=tf.int32).numpy()
    confidences = tf.reduce_max(probs, axis=1).numpy()
    return preds, confidences
