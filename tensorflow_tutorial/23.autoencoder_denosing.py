# Setting
# 1. import library.
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy.random as rnd
import numpy as np
import pathlib
import sys
import os

# 2. set directory.
PROJECT_ROOT_DIR = ''
CHAPTER_UID = 'autoencoder'

model_save_dir = os.path.join(PROJECT_ROOT_DIR, "model", CHAPTER_UID)
image_save_dir = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_UID)
log_save_dir = os.path.join(PROJECT_ROOT_DIR, "logs", CHAPTER_UID)

pathlib.Path(model_save_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(image_save_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(log_save_dir).mkdir(parents=True, exist_ok=True)

# 3. set hyper-parameter.
# input , hidden , output neuron
h = 28
w = 28
n_inputs = h * w
n_hiddens1 = 300
n_hiddens2 = 150
n_hiddens3 = n_hiddens1
n_outputs = n_inputs

# learning rate, regulerization, epoch, batch-size, initializer
learning_rate = 0.003
n_epochs = 4
batchsize = 150
l2_reg = 0.0009
noise_level = 0.01
reguralization = tf.contrib.layers.l2_reguraliztion(l2_reg)
initializer = tf.global_variables_initializer()


# 4. create reset_graph function.
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    rnd.seed(seed)


# 5. create save_fig function.
def save_fig(fig_id, tight_layout=True):
    path = os.path.join(image_save_dir, fig_id + ".png")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=".png", dpi=300)


# 6. create plot_image function.
def plot_image(image, shape=[h, w]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")


# 7. create show_reconstruction_digit function.
def show_reconstruction_digits(X, outputs, model_path=None, n_test_digits=2):
    with tf.Session as sess:
        if model_path:
            saver.restore(sess, model_path)
        outputs_val = outputs.eval(feed_dict={X: X_test[:n_test_digits]})

    fig = plt.figure(figsize=(8, 3 * n_test_digits))
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index + 2 + 1)
        plot_image(X_test[digit_index])
        plt.subplot(n_test_digits, 2, digit_index + 2 + 2)
        plot_image(outputs_val[digit_index])


# 8. create shuffle_batch function.
def shuffle_batch(X, y, batchsize):
    rnd_index = rnd.permutation(len(X))
    n_batches = len(X) // batchsize
    for batch_index in np.array_split(rnd_index, n_batches):
        X_batch, y_batch = X[batch_index], y[batch_index]
        yield X_batch, y_batch

# 1
# input data.
X_train, y_train, X_test, y_test = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, n_inputs)/255.0
y_train = y_train.astype(np.int32)
X_test = X_test.astype(np.float32).reshape(-1, n_inputs)/255.0
y_test = y_test.astype(np.int32)

X_valid, y_valid = X_train[:50000], y_train[:50000]
X_train, y_train = X_train[50000:], y_train[50000:]

# 2
# create model.
reset_graph()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
X_noisy = X + noise_level * tf.random_normal(tf.shape(X))

hidden1 = tf.layers.dense(X_noisy, n_hiddens1, activation=tf.nn.relu, name="hidden1")
hidden2 = tf.layers.dense(hidden1, n_hiddens2, activation=tf.nn.relu, name="hidden2")
hidden3 = tf.layers.dense(hidden2, n_hiddens3, activation=tf.nn.relu, name="hidden3")
outputs = tf.layers.dense(hidden3, n_outputs, name="outputs")

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(reconstruction_loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# 2
# run train.
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = len(X_train) // batchsize
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            X_batch, y_batch = next(shuffle_batch(X_train, y_train, batchsize))
            sess.run(training_op, feed_dict={X: X_batch})
        loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
        print("\r{}".format(epoch), "훈련 MSE:", loss_train)
        saver.save(sess, os.path.join(model_save_dir, "my_model_stacked_denoising_gausian.ckpt"))
