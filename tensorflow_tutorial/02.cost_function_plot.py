import tensorflow as tf
import matplotlib.pyplot as plt


# 1
# Build graph using TF operations.
X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * w
hypothesis = X * W

# cost function.
cost = tf.reduce_mean(tf.square(hypothesis - Y))


# 2,3
# Run/update graph and get results
# Launch the graph in a session
sess = tf.Session()

# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Variables for plotting cost function
W_val = []
cost_val = []
for i in range(-30, 50):
    # w는 -3 ~ -5사이의 값을 가짐.
    feed_W = i * 0.1
    # w가 변하면서 cost가 어떻게 변하는지 보는 그래프.
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

# show the cost function.
plt.plot(W_val, cost_val)
plt.show()

