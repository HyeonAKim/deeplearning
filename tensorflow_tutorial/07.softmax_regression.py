"""
softmax function : n 개의 예측을 할때 사용
- 왜 소프트맥스를 사용하는가? 모든 레이블클래스를 확률로 표현이 가능함 다시말해 0-1 사이의 값으로 할 수 있음
- 어떻게 구현 하는가?
hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)


softmax에 걸맞는 cost function은? cross entropy
- 어떻게 계산하는가? 각 레이블에서 0 - 1 사이의 값을 뒀을 때의 차이를 더함
- 어떻게 구현하는가?
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1)
optimizer = tf.train.GradientDescentOpimizer(learning_rate=0.1).minizer(cost)

"""
import tensorflow as tf

# 1
# Input data
x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5],
          [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
# one-hot encoding
y_data = [[0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0],
          [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]

X = tf.placeholder("float32", [None, 4])
Y = tf.placeholder("float32", [None, 3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# 2
# Set activation function : SoftMax
# SoftMax = exp(Logits) / reduce_sum(exp(Logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# 3
# Set cost function : cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# 4
# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        # run optimizer
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X : x_data, Y: y_data}))

    # 5
    # Testing & One-hot encoding
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9],
                                            [1, 3, 4, 3]]})
    print(a, sess.run(tf.argmax(a, 1)))
