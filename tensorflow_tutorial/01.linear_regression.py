import tensorflow as tf

# 1
# Build graph using TF operations.
# X and Y data
# x_train = [1, 2, 3]
# y_train = [1, 2, 3]

# palce holder 를 사용하는 이유는 모델을 생성하고 값을 전달할 수 있기 때문
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# 텐서플로우가 학습에서 변경하는 값
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Our hypothesis XW+b
# hypothesis = x_train * W + b
hypothesis = X * W + b

# cost/Loss function
# cost = tf.reduce_mean(tf.square(hypothesis - y_train))
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)


# 2,3
# Run/update graph and get results
# Launch the graph in a session
sess = tf.Session()

# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Fit the line
# for step in range(2001):
#     sess.run(train)
#     if step % 20 == 0:
#         print(step, sess.run(cost), sess.run(W), sess.run(b))

for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                                         feed_dict={X: [1, 2, 3, 4, 5],
                                                    Y: [2.1, 3.1, 4.1, 5.1, 6.1]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)


# 4
# Testing our model
print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))