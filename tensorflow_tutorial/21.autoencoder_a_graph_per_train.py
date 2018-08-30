# Setting
# 0. Import libraries.
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import numpy.random as rnd
import pathlib
import sys
import os


# 1. Set hyper-parameter.
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01
epoch_step_1 = 4
epoch_step_2 = 10
batch_size = 150
l2_reg = 0.0001

activation = tf.nn.elu
regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
initializer = tf.variance_scaling_initializer()


# 2. Set save directory.
PROJECT_ROOT_DIR = "/home/wisemold/Python/old/test_hyeona"
CHAPTER_ID = "autoencoder"

save_model_dir = os.path.join(PROJECT_ROOT_DIR, "model")
save_image_dir = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
save_log_dir = os.path.join(PROJECT_ROOT_DIR, "logs", CHAPTER_ID)
pathlib.Path(save_model_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(save_image_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(save_log_dir).mkdir(parents=True, exist_ok=True)


# 3. Set random seed.
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# 4. Save fig.
def save_fig(fig_id, tight_layout=True):
    path = os.path.join(save_image_dir, fig_id+'.png')
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


# 5. Plot image.
def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")


# 6. Show reconstruct fig.
def show_reconstruct_digits(X, outputs, model_path=None, n_test_digits=2):
    with tf.Session() as sess:
        if model_path:
            saver.restore(sess, model_path)
        outputs_val = outputs.eval(feed_dict={X: X_test[:n_test_digits]})

    fig = plt.figure(figsize=(8 , 3 * n_test_digits))
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index  * 2 + 1)
        plot_image(X_test[digit_index])
        plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
        plot_image(outputs_val[digit_index])

# 1
# Input data.
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
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

# 2
# Create model graph.
reset_graph()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

weight1_init = initializer([n_inputs, n_hidden1])
weight2_init = initializer([n_hidden1, n_hidden2])
weight3_init = initializer([n_hidden2, n_hidden3])
weight4_init = initializer([n_hidden3, n_outputs])

weights1 = tf.Variable(weight1_init, dtype=tf.float32, name="weights1")
weights2 = tf.Variable(weight2_init, dtype=tf.float32, name="weights2")
weights3 = tf.Variable(weight3_init, dtype=tf.float32, name="weights3")
weights4 = tf.Variable(weight4_init, dtype=tf.float32, name="weights4")

biases1 = tf.Variable(tf.zeros(n_hidden1), name="biases1")
biases2 = tf.Variable(tf.zeros(n_hidden2), name="biases2")
biases3 = tf.Variable(tf.zeros(n_hidden3), name="biases3")
biases4 = tf.Variable(tf.zeros(n_outputs), name="biases4")

hidden1 = activation(tf.matmul(X, weights1) + biases1)
hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
hidden3 = activation(tf.matmul(hidden2, weights3) + biases3)
outputs = tf.matmul(hidden3, weights4) + biases4

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
optimizer = tf.train.AdamOptimizer(learning_rate)


# 3
# Create train scope.
with tf.name_scope("phase1"):
    phase1_outputs = tf.matmul(hidden1, weights4) + biases4
    phase1_reconstruction_loss = tf.reduce_mean(tf.square(phase1_outputs - X))
    phase1_reg_loss = regularizer(weights1) + regularizer(weights4)
    phase1_loss = phase1_reconstruction_loss + phase1_reg_loss
    phase1_training_op = optimizer.minimize(phase1_loss)

with tf.name_scope("phase2"):
    phase2_reconstruction_loss = tf.reduce_mean(tf.square(hidden3 - hidden1))
    phase2_reg_loss = regularizer(weights2) + regularizer(weights3)
    phase2_loss = phase2_reconstruction_loss + phase2_reg_loss
    train_vars = [weights2, biases2, weights3, biases3]
    phase2_training_op = optimizer.minimize(phase2_loss, var_list=train_vars)  # hidden 1 동결


# 4
# Run train.
init = tf.global_variables_initializer()
saver = tf.train.Saver()

training_ops = [phase1_training_op, phase2_training_op]
reconstruction_losses = [phase1_reconstruction_loss, phase2_reconstruction_loss]
n_epochs = [epoch_step_1, epoch_step_2]
batch_sizes = [batch_size, batch_size]

with tf.Session() as sess:
    init.run()
    for phase in range(2):
        print("훈련단계 #{}".format(phase + 1))
        for epoch in range(n_epochs[phase]):
            n_batches = len(X_train) // batch_sizes[phase]
            for iteration in range(n_batches):
                print("\r{}%".format(100 * iteration // n_batches), end="")
                sys.stdout.flush()
                X_batch, y_batch = next(shuffle_batch(X_train, y_train, batch_sizes[phase]))
                sess.run(training_ops[phase], feed_dict={X: X_batch})
            loss_train = reconstruction_losses[phase].eval(feed_dict={X: X_batch})
            print("\r{}".format(epoch), "훈련 MES:", loss_train)
            saver.save(sess,  os.path.join(save_model_dir, "epoch_"+str(n_epochs[0])+"_"+str(n_epochs[1])+"_lr_"+str(learning_rate)+"my_model_one_at_a_time.ckpt"))
        loss_test = reconstruction_loss.eval(feed_dict={X: X_test})
        print("테스트 MSE:", loss_test)

# 5
# Check Test images and save reconstruction image.

show_reconstruct_digits(X, outputs, os.path.join(save_model_dir, "epoch_"+str(n_epochs[0])+"_"+str(n_epochs[1])+"_lr_"+str(learning_rate)+"my_model_one_at_a_time.ckpt"))
save_fig("reconstruction_one_at_a_time_plot_"+str(n_epochs[0])+"_"+str(n_epochs[1])+'_'+str( learning_rate ))