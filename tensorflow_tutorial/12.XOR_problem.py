"""
XOR : 하나의 유닛으로는 절대로 풀 수 없는 문제
But 2개 , 3개의 유닛으로는 풀 수 있음
0, 0 > 0
0, 1 > 1
1, 0 > 1
1, 1 > 0

x = [[0, 0], [0, 1], [1, 0], [1, 1]]
k(x) = sigmoid(XW1 + B1)
Y = H(x) = sigmoid(K(x)W2 + B2)

# NN
K = tf.sigmoid(tf.matmul(X, W1) + b1)
hypothesis = tf.sigmoid(tf.matmul(K, W2) + b2)

# 어떻게 W, b를 학습할까?
- cost 함수 그래프를 그릴려면 미분을 알아야함
- 노드가 많아지면서 미분값 구하는게 복잡해짐 > 구할 수 없다!
- 1974, 1986 : backpropagation hinton 교수에 의해 해결됨
- Back propagation ( chain rule )
"""
import numpy as np
import tensorflow as tf

# 1
# Input Data

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 1 layer
W1 = tf.Variable(tf.random_normal([2, 10]), name='weight1')
b1 = tf.Variable(tf.random_normal([1]), name='bias1')
layer_1 = tf.sigmoid(tf.matmul(X, W1)+b1)

# 2 layer
W2 = tf.Variable(tf.random_normal([10, 10]), name='weight2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
layer_2 = tf.sigmoid(tf.matmul(layer_1, W2)+b2)

# 3 layer
W3 = tf.Variable(tf.random_normal([10, 10]), name='weight3')
b3 = tf.Variable(tf.random_normal([1]), name='bias3')
layer_3 = tf.sigmoid(tf.matmul(layer_2, W3)+b3)

# 4 layer
W4 = tf.Variable(tf.random_normal([10, 1]), name='weight4')
b4 = tf.Variable(tf.random_normal([1]), name='bias4')

hypothesis = tf.sigmoid(tf.matmul(layer_3, W4) + b4)

# 3
# Set cost function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y)*tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# 4
# Accuracy computation
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.int32))

# 5
# Lauch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run([W1, W2]))

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\n Hypothesis: ", h, "\n Correct:", c, "\n Accuracy: ", a)
