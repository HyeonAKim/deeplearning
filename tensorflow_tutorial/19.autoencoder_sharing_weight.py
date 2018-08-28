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
plt.rcParams
# Create save fig function : save_fig()
# Create plot image function : plot_image(), plot_multiple_imges()

# 1
# Set hyper-parameter
learning_rate = 0.01
l2_reg = 0.0001
n_epochs = 5
batch_size = 150


# 2
# Design model
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150
n_hidden3 = n_hidden1
n_outputs = n_inputs

activation = tf.nn.relu
reqularizer = tf.contrib.layers.l2_regularize(l2_reg)
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
reg_losses = reqularizer(weights1) + reqularizer(weights2)
loss = [reconstruction_loss] + reg_losses

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

# 5
# Initialize values and Run train.
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = len(X_train)  # Batch size
        for iteration in range(n_batches):
            # suffle_batch()
            X_batch, y_batch = next(shuffle_batch(X_train, y_train, batch_size))
            sess.run(training_op, feed_dict={X: X_batch})

