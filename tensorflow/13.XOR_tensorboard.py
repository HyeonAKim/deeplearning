"""
Visualize yout TF graph
Plot quantitative metrics
Show additional data

5- Step
1. From TF graph, decide which tensors you want to log
w2_hist = tf.summary.histogram("weight2", w2)
cost_sum = tf.summary.scalar("cost2", cost)

2. Merge all summaries
summary = tf.summary.merge_all()

3. Create writer and add graph
writer = tf.summary.FileWriter('./logs')
writer.add_graph(sess.graph)

4. Run summary merge and add_summary
s, _ = sess.run([summary, opimizer], feed_fict=feed_dict)
writer.add_summary(s, global_step = global_step)

5. Lauch Tensorboard
커맨드창에서 텐서보드 실행
tensorboard --logdir=./logs
tensorboard --host=127.0.0.1 --logdir=C:/Users/HyunA/PycharmProjects/deeplearning/tensorflow/logs
"""
import numpy as np
import tensorflow as tf

# 1
# Input Data

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 그래프그릴때 레이어별로 깔끔하게 정리해주는 역할
with tf.name_scope("layer1") as scope:
    # 1 layer
    W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
    b1 = tf.Variable(tf.random_normal([1]), name='bias1')
    layer_1 = tf.sigmoid(tf.matmul(X, W1) + b1)
    # 하나의 값이 아닐경우 histogram 으로 볼 수 있음.
    w1_hist = tf.summary.histogram( "weight1", W1 )
    b1_hist = tf.summary.histogram( "bias1", b1 )
    layer_1_hist = tf.summary.histogram( "layer_1", layer_1 )

with tf.name_scope("layer2") as scope:
    # 2 layer
    W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
    b2 = tf.Variable(tf.random_normal([1]), name='bias2')
    hypothesis = tf.sigmoid(tf.matmul(layer_1, W2)+b2)

    # 하나의 값이 아닐경우 histogram 으로 볼 수 있음.
    w2_hist = tf.summary.histogram("weight2", W2)
    b2_hist = tf.summary.histogram("bias2", b2)
    hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)

# 3
# Set cost function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y)*tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
cost_summ = tf.summary.scalar("cost", cost)

# 4
# Accuracy computation
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.int32))
accuracy_summ = tf.summary.scalar("accuracy", accuracy)

# 5
# Tensor Board summary
summary = tf.summary.merge_all()

# 6
# Lauch graph
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./logs/logs_acc')
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        s, _, _ = sess.run([summary, accuracy, train], feed_dict={X: x_data, Y: y_data})
        writer.add_summary(s, global_step=step)