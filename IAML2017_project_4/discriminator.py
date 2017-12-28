import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.python.ops.rnn_cell import BasicRNNCell, BasicLSTMCell, GRUCell


class Discriminator(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size, hidden_dim, num_layers, embedding_size):  # , filter_sizes, num_filters, l2_reg_lambda=0.0
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.variable_scope('discriminator'):
            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="W")
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            x = tf.unstack(self.embedded_chars, sequence_length, 1)
            stack_rnn = []
            for i in range(num_layers):
                with tf.name_scope("lstm-%s" % (i + 1)):
                    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
                    cell = tf.contrib.rnn.AttentionCellWrapper(cell, 5)
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=self.dropout_keep_prob)
                    stack_rnn.append(cell)
            stacked_cell = tf.nn.rnn_cell.MultiRNNCell(stack_rnn, state_is_tuple=True)
            outputs, last_states = rnn.static_rnn(stacked_cell, x, dtype=tf.float32)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.Variable(tf.truncated_normal([hidden_dim, num_classes], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                self.scores = tf.nn.xw_plus_b(outputs[-1], W, b, name="scores")
                self.ypred_for_auc = tf.nn.softmax(self.scores)
                self.predictions = tf.argmax(self.scores, 1, name="predictions")

            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses)  # + l2_reg_lambda * l2_loss

        self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
        d_optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = d_optimizer.apply_gradients(grads_and_vars)
        correct_prediction = tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
