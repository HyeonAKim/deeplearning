"""
Stacked Autoencoder
"""
# Setting
from __future__ import division, print_function, unicode_literals
from functools import partial
import tensorflow as tf
import numpy as np
import pathlib
import os
import sys

# Reset graph
def reset_graph(seed = 42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# Setting matplotlib
import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

plt.rcParams['font.family'] = 'NanumBarunGothic'
plt.rcParams['axes.unicode_minus'] = False


# Create save fig function : save_fig()
PROJECT_ROOT_DIR = "C:\\Users\\HyunA\\PycharmProjects\\deeplearning\\tensorflow_tutorial"
CHAPTER_ID = "autoencoder"


def save_fig(fig_id, tight_layout=True):
    dir_path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
    path = os.path.join(dir_path, fig_id+'.png')
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

# Create plot image function : plot_image(), plot_multiplt_image()

def plot_image(image, shape = [28, 28]):
    plt.imshow(image.reshape(shape) , cmap="Greys", interpolation='nearest')
    plt.axis("off")


def plot_multiple_images(images, n_rows, n_cols, pad=2):
    images = images - images.min()
    w, h = images[1:]
    image = np.zeros((w + pad) * n_rows + pad, (h + pad) * n_cols + pad)
    for y in range(n_rows):
        for x in range(n_cols):
            image[(y*(h+pad)+pad):(y*(h+pad)+pad+h), (x*(w+pad)+pad):(x*(w+pad)+pad+w)] = images[y * n_cols + x]
    plt.imshow(image, cmap="Greys", interpolation="nearest")
    plt.axis("off")


# 1
# Input data.
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[: 5000], X_train[5000:]
y_valid, y_train = y_train[: 5000], y_train[5000:]


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X)  # batch size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

# Design model
reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150
n_hidden3 = n_hidden1
n_outputs = n_inputs

# 2
# Set hyper-parameter
learning_rate = 0.01
l2_reg = 0.0001
n_epochs = 5
batch_size = 150

# 3
# Create model
X = tf.placeholder(tf.float32, shape=[None, n_inputs])
he_init = tf.variance_scaling_initializer()
l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)

my_dense_layer = partial(tf.layers.dense,
                         activation=tf.nn.relu,
                         kernel_initializer=he_init,
                         kernel_regularizer=l2_regularizer)

hidden1 = my_dense_layer(X, n_hidden1)
hidden2 = my_dense_layer(hidden1, n_hidden2)
hidden3 = my_dense_layer(hidden2, n_hidden3)
outputs = my_dense_layer(hidden3, n_outputs, activation=None)

# 4
# Set loss function and optimizer.
reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))  # MSE
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses)
recostrcuction_loss_summ = tf.summary.scalar("reconstrcution_loss", reconstruction_loss)
loss_summ = tf.summary.scalar("loss", loss)
summary = tf.summary.merge_all()

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

# 5
# Initialize values and Run train.
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.path.join(PROJECT_ROOT_DIR,'/logs/autoencoder/train'))
    writer.add_graph(sess.graph)

    init.run()
    for epoch in range(n_epochs):
        n_batches = len(X_train)  # Batch size
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            # shuffle_batch()
            X_batch, y_batch = next(shuffle_batch(X_train, y_train, batch_size))
            s, _ = sess.run([summary, training_op], feed_dict={X: X_batch})
            writer.add_summary(s, global_step=iteration)
        loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
        print("\r{}".format(epoch), "훈련 MES:", loss_train)
        saver.save(sess, "./my_model_all_layers.ckpt")


# 6
# load model and validation model.

def show_reconstructed_digits(X, outputs, model_path=None, n_test_digits = 2):
    with tf.Session() as sess:
        if model_path:
            saver.restore(sess, model_path)
        output_val = outputs.eval(feed_dict={X: X_test[:n_test_digits]})

    fig = plt.figure(figsize=(8, 3 * n_test_digits ))
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
        plot_image(X_test[digit_index])
        plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
        plot_image(output_val[digit_index])


show_reconstructed_digits(X, outputs, "./my_model_all_layers.ckpt")
save_fig("reconstruction_plot")