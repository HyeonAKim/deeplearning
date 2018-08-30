# Setting
# 1. Import libries.
import tensorflow as tf
import numpy as np
import numpy.random as rnd

import os
import sys
import pathlib


# 2. set random seed.
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# 3. save fig.
import matplotlib.pyplot as plt
PROJECT_ROOT_DIR = "/home/wisemold/Python/old/test_hyeona"
CHAPTER_ID = "autoencoder"


def save_fig(fig_id, tight_layout=True):
    dir_path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
    path = os.path.join(dir_path, fig_id+'.png')
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


# 3. plot images.
def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")


# 4. show reconstruct fig.
def show_reconstructed_digits(X, outputs, model_path=None, n_test_digits=2):
    with tf.Session() as sess:
        if model_path:
            saver.restore(sess, model_path)
        outputs_val = outputs.eval(feed_dict={X: X_test[:n_test_digits]})

    fig = plt.figure(figsize=(8, 3 * n_test_digits))
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
        plot_image(X_test[digit_index])
        plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
        plot_image(outputs_val[digit_index])

# 1
# Input data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:50000], X_train[50000:]
y_valid, y_train = y_train[:50000], y_train[50000:]


def shuffle_batch(X, y, batchsize):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batchsize
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

# 2
# Create model
reset_graph()

from functools import partial


n_epoch = 4
learning_rate = 0.01

def train_autoencoder(X_train, n_neurons, n_epochs, batch_size,
                      learning_rate = 0.01, l2_reg = 0.0005, seed=42,
                      hidden_activation=tf.nn.elu,
                      output_activation=tf.nn.elu):
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(seed)
        n_inputs = X_train.shape[1]
        X = tf.placeholder(tf.float32, shape=[None, n_inputs])

        my_dense_layer = partial(
            tf.layers.dense,
            kernel_initializer = tf.variance_scaling_initializer(),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(l2_reg))

        hidden = my_dense_layer(X, n_neurons, activation=hidden_activation, name="hidden")
        outputs = my_dense_layer(hidden, n_inputs, activation=output_activation, name="outputs")

        reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([reconstruction_loss]+reg_losses)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        init.run()
        for epoch in range(n_epochs):
            n_batches = len(X_train)//batch_size
            for iteration in range(n_batches):
                print("\r{}%".format(100*iteration//n_batches), end="")
                sys.stdout.flush()
                indices = rnd.permutation(len(X_train))[:batch_size]
                X_batch = X_train[indices]
                sess.run(training_op, feed_dict={X: X_batch})
            loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
            print("\r{}".format(epoch), "훈련 MSE:", loss_train)
        params = dict( [(var.name, var.eval()) for var in tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES )] )
        hidden_val = hidden.eval( feed_dict={X: X_train} )
        return hidden_val, params["hidden/kernel:0"], params["hidden/bias:0"], params["outputs/kernel:0"], params[
            "outputs/bias:0"]

# 3
# Train autoencoder. - 2 layers

hidden_output, W1, b1, W4, b4 = train_autoencoder(X_train, n_neurons=300, n_epochs = n_epoch, batch_size=150, output_activation=None, learning_rate=learning_rate)
_, W2, b2, W3, b3 = train_autoencoder(hidden_output, n_neurons=150, n_epochs=n_epoch, batch_size=150, learning_rate=learning_rate)

# 4
# create strack autoencoder - using trained weight, bias
reset_graph()
n_inputs = 28 * 28

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden1 = tf.nn.elu(tf.matmul(X, W1) + b1)
hidden2 = tf.nn.elu(tf.matmul(hidden1, W2) + b2)
hidden3 = tf.nn.elu(tf.matmul(hidden2, W3) + b3)
outputs = tf.matmul(hidden3, W4) + b4


show_reconstructed_digits(X, outputs)
save_fig("reconstruction_layer_plot"+str(n_epoch)+'_'+str(learning_rate))