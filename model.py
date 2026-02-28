import tensorflow as tf
import numpy as np
import logging
import meta

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
        out = tf.sparse.sparse_dense_matmul(adj_norm, xw)
        return self.activation(out + self.bias)

class GCNClassifier(tf.keras.Model):
    def __init__(self, hidden_dim=meta.hidden_dim, num_classes=2):
        super().__init__()
        self.gcn1 = GCNLayer(hidden_dim, activation="relu")
        self.gcn2 = GCNLayer(hidden_dim, activation="relu")
        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, x, adj_norm):
        h = self.gcn1(x, adj_norm)
        h = self.gcn2(h, adj_norm)
        return self.classifier(h)

def _normalize_adjacency(adj):
    n = adj.dense_shape[0]

    self_loops = tf.SparseTensor(
        indices=tf.stack([tf.range(n, dtype=tf.int64),
                          tf.range(n, dtype=tf.int64)], axis=1),
        values=tf.ones(n, dtype=tf.float32),
        dense_shape=adj.dense_shape,
    )
    adj_hat = tf.sparse.add(adj, self_loops)

    degree = tf.sparse.reduce_sum(adj_hat, axis=1)
    d_inv_sqrt = tf.where(degree > 0, tf.pow(degree, -0.5), 0.0)

    row = adj_hat.indices[:, 0]
    col = adj_hat.indices[:, 1]
    scaled_values = adj_hat.values * tf.gather(d_inv_sqrt, row) * tf.gather(d_inv_sqrt, col)

    return tf.SparseTensor(
        indices=adj_hat.indices,
        values=scaled_values,
        dense_shape=adj_hat.dense_shape,
    )


def train(node_features, adjacency, labels,
          hidden_dim=meta.hidden_dim, num_classes=None, epochs=meta.epochs, lr=0.01):
    labels = tf.cast(tf.constant(labels), tf.int32)
    if num_classes is None:
        num_classes = int(tf.reduce_max(labels).numpy()) + 1
    adj_norm = _normalize_adjacency(adjacency)
    adj_norm = tf.sparse.reorder(adj_norm)

    model = GCNClassifier(hidden_dim=hidden_dim, num_classes=num_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for epoch in range(1, epochs + 1):
        with tf.GradientTape() as tape:
            logits = model(node_features, adj_norm)
            loss = loss_fn(labels, logits)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(preds, labels), tf.float32)
        )
        logging.info(
            f"Epoch {epoch:>3d}/{epochs}  loss={loss.numpy():.4f}  "
            f"accuracy={accuracy.numpy():.4f}"
        )

    return model


def predict(model, node_features, adjacency):
    """
    Run inference on a graph and return predicted class indices.

    Args:
        model: A trained GCNClassifier.
        node_features: tf.Tensor of shape [N, F].
        adjacency: tf.SparseTensor of shape [N, N].

    Returns:
        1-D numpy array of predicted class indices, shape [N].
    """
    adj_norm = _normalize_adjacency(adjacency)
    adj_norm = tf.sparse.reorder(adj_norm)
    logits = model(node_features, adj_norm)
    return tf.argmax(logits, axis=1, output_type=tf.int32).numpy()
