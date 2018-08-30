"""
오토인코더가 완벽하게 대칭일때 인코더와 디코더의 가중치를 묶어서 사용
이점1.
    훈력 속도를 높일 수 있음
이점2.
    과대적합의 위험을 줄여줌.

중요한 사실
    1. weight3 와 weights 4는 변수로 선언되지 않았고 각각 weights2dhk weights1 의 전치
    2. 변수가 아니기 때문에 규제에 사용하지 않음
    3. 편향은 묶지도 않고 규제도 하지 않음.
"""
# Setting
from __future__ import division, print_function, unicode_literals
from functools import partial
import tensorflow as tf
import numpy as np
import os
import pathlib
import sys


# Create reset_graph function : reset_graph()
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# Set matplotlib
import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

plt.rcParams['font.family'] = 'NanumBarunGothic'
plt.rcParams['axes.unicode_minus'] = False

# Create save fig function : save_fig()
PROJECT_ROOT_DIR = "/home/wisemold/Python/old/test_hyeona"
CHAPTER_ID = "autoencoder"


def save_fig(fig_id, tight_layout=True):
    dir_path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
    path = os.path.join(dir_path, fig_id+'.png')
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


# Create plot image function : plot_image(), plot_multiple_imges()
def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation='nearest')
    plt.axis("off")


def plot_multiple_images(images, n_rows, n_cols, pad=2):
    images = images - min(images)
    w, h = images[1:]
    image = np.zeros((w + pad) * n_rows + pad, (h + pad) * n_cols + pad)
    for y in range(n_rows):
        for x in range(n_cols):
            image[(y*(h+pad)+pad):(y*(h+pad)+pad+h), (x*(w+pad)+pad):(x*(w+pad)+pad+w)] = images[y * n_cols + x]
    plt.imshow(image, cmap="Greys", interpolation='nearest')
    plt.axis("off")


# reconstructed_digits
def show_reconstructed_digits(X, outputs, model_path=None, n_test_digits=2):
    with tf.Session() as sess:
        if model_path:
            saver.restore(sess, model_path)
        outputs_val = outputs.eval(feed_dict={X: X_test[:n_test_digits]})

    fig = plt.figure(figsize=(8, 3*n_test_digits))
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
        plot_image(X_test[digit_index])
        plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
        plot_image(outputs_val[digit_index])


# 1
# Set hyper-parameter
reset_graph()
learning_rate = 0.01
l2_reg = 0.0001
n_epochs = 5
batch_size = 150

2
# Input MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]


def shuffle_batch(X, y, batchsize):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

# 2
# Design model
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150
n_hidden3 = n_hidden1
n_outputs = n_inputs

activation = tf.nn.relu
regularizer = tf.contrib.layers.l2_regularize(l2_reg)
initializer = tf.variance_scaling_initializer()

# 3
# Create model

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

weights1_init = initializer([n_inputs, n_hidden1])
weights2_init = initializer([n_hidden1, n_hidden2])

weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weight1")
weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weight2")
weights3 = tf.Variable(weights2, dtype=tf.float32, name="weight3")  # 가중치 묶기
weights4 = tf.Variable(weights1, dtype=tf.float32, name="weight4")  # 가중치 묶기

biases1 = tf.Variable(tf.zeros(n_hidden1), name="biases1")
biases2 = tf.Variable(tf.zeros(n_hidden2), name="biases2")
biases3 = tf.Variable(tf.zeros(n_hidden3), name="biases3")
biases4 = tf.Variable(tf.zeros(n_outputs), name="biases4")

hidden1 = activation(tf.matmul(X, weights1) + biases1)
hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
hidden3 = activation(tf.matmul(hidden2, weights3) + biases3)
outputs = tf.matmul(hidden3, weights4) + biases4

# 4
# Set loss function and optimizer.
reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))  # MSE
reg_losses = regularizer(weights1) + regularizer(weights2)
loss = [reconstruction_loss] + reg_losses
recostrcuction_loss_summ = tf.summary.scalar("reconstrcution_loss", reconstruction_loss)
loss_summ = tf.summary.scalar("loss", loss)
summary = tf.summary.merge_all()

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

# 5
# Initialize values and Run train.
model_path = os.path.join('/home/wisemold/Python/old/test_hyeona/model')
pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.path.join( PROJECT_ROOT_DIR, 'logs', 'autoencoder', "sharing_weight" + str( learning_rate ) ) )
    writer.add_graph( sess.graph )

    init.run()
    for epoch in range(n_epochs):
        n_batches = len(X_train) // batch_size
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            X_batch, y_batch = next(shuffle_batch(X_train, y_train, batch_size))
            s, _ = sess.run([summary, training_op], feed_dict={X: X_batch})
            writer.add_summary( s, global_step=iteration )
        loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
        print("\r{}%".format(epoch), "훈련 MSE:", loss_train)
        saver.save( sess, os.path.join( model_path, "my_model_tying_weights.ckpt" ) )

show_reconstructed_digits(X, outputs,  os.path.join( model_path, "my_model_tying_weights.ckpt" ))
save_fig("reconstruction_weight_sharing_plot")