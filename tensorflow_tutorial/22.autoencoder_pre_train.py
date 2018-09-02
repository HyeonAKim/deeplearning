# Setting
# 1. import library
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import numpy.random as rnd
import pathlib
import os
import sys

# 2. setting directory
PROJECT_ROOT_DIR = "/home/wisemold/Python/old/test_hyeona"
CHAPTER_ID = "autoencoder"

model_save_dir = os.path.join(PROJECT_ROOT_DIR, "model")
image_save_dir = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
log_save_dir = os.path.join(PROJECT_ROOT_DIR, "logs", CHAPTER_ID )

pathlib.Path(model_save_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(image_save_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(log_save_dir).mkdir(parents=True, exist_ok=True)

# 3. setting hyper-parameter
# input size & neuron option per layers
img_h = 28
img_w = 28
n_inputs = img_h * img_w
n_hidden1 = 300
n_hidden2 = 150
n_outputs = n_inputs

# learing_rate, epoch, batchsize, l2_reg
learning_rate = 0.003
n_epoch = 20
batchsize = 150
l2_reg = 0.0001

# activation function , regularizer , initializer
activation = tf.nn.elu
regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
initializer = tf.variance_scaling_initializer()


# 4. reset graph
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    rnd.seed(seed)


# 5. save fig
def save_fig(fig_id, tight_layout=True):
    path = os.path.join(image_save_dir, fig_id+".png")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


# 6. plot image
def plot_image(image, shape=[h, w]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axes("off")


# 7. show reconstruction image
def show_reconstrunction_digits(X, outputs, model_path=None, n_test_digits=2):
    with tf.Session as sess:
        if model_path:
            saver.restore(sess, model_path)
        output_val = outputs.eval(feed_dict={X: X_test[ :n_test_digits]})

    fig = plt.figure(figsize=(8, 3 * n_test_digits))
    for digit_idx in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_idx * 2 + 1)
        plot_image(X_test[digit_idx])
        plt.subplot(n_test_digits, 2, digit_idx * 2 + 2)
        plot_image(X_test[digit_idx])


# 8. input data
X_train, y_train, X_test, y_test = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, n_inputs) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, n_inputs) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

X_valid, X_train = X_train[:50000], X_train[50000:]
y_valid, y_train = y_train[:50000], y_train[50000:]


def shuffle_batch(X, y, batchsize):
    rnd_idx = rnd.permutation(len(X))
    n_batches = len(X) // batchsize
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

# 9. create model
reset_graph()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
y = tf.placeholder(tf.int32, shape=[None])

weights1_init = initializer([n_inputs, n_hidden1])
weights2_init = initializer([n_hidden1, n_hidden2])
weights3_init = initializer([n_hidden2, n_outputs])

weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
weights3 = tf.Variable(weights3_init, dtype=tf.float32, name="weights3")

biases1 = tf.Variable(tf.zeros(n_hidden1), name="biases1")
biases2 = tf.Variable(tf.zeros(n_hidden2), name="biases2")
biases3 = tf.Variable(tf.zeros(n_outputs), name="biases3")

hidden1 = activation(tf.matmul(X, weights1) + biases1)
hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
logits = tf.matmul(hidden2, weights3) + biases3

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(la)
reg_loss = regularizer(weights1) + regularizer(weights2) + regularizer(weights3)
loss = cross_entropy + reg_loss
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, var_list=[weights3, biases3])  # layer 1과 2를 동결

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.case(correct, tf.float32))

init = tf.global_variables_initializer()
pretrain_saver = tf.train.Saver([weights1, weights2, weights3])
saver = tf.train.Saver()

# 10. run train
n_labeled_instance = 20000

with tf.Session() as sess:
    init.run()
    pretrain_saver.restore(sess, os.path.join(model_save_dir, "my_model_cache_frozen.ckpt"))
    for epoch in range(n_epoch):
        n_batches = n_labeled_instance // batchsize
        for iteration in range(batchsize):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            indices = rnd.permutation(n_labeled_instance)[:batchsize]
            X_batch, y_batch = X_train[:indices], y_train[:indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        print("\r{}".format(epoch), "검증세트 정확도 : ", accuracy_val, end="\t")
        saver.save(sess, os.path.join(model_save_dir, "my_model_supervised.ckpt"))
        test_val = accuracy.val(feed_dict={X: X_test, y: y_test})
        print("테스트 정확도 : ", test_val)
