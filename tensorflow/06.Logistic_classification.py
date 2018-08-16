"""
Regression
- 기본적인 hyputhesis : 선형 함수
- cost 함수 : (예측값 - 실제값) 제곱평균
- Gradient decent : 경사(미분)가 어디를 향해있는지 보고 낮은 점으로 이동(학습률 learning rate)

Binary Classification
- hypothesis : 시그모이드 함수 (0 - 1 사이의 값)
- cost 함수 :  (예측값 - 실제값) 제곱의 평균 - local minimun 에 빠질 수 있음 >  cost 함수를 변경해야함
           :  예측값에 1 ; -log(H(x)) / 0: -log(1-H(x)) > cost 함수를 convex 하게 만들어 줌
- Gradient decent algorithm : cost 함수 미분에 따라서 낮은점으로 이동

# cost function
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis)))

# Minimize
a = tf.Variable(0.1) # learingrate
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)
"""
import numpy as np
import tensorflow as tf

# 1
# Data
# x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
# y_data = [[0], [0], [0], [1], [1], [1]]

# xy = np.loadtxt('/Users/hyeonakim/PycharmProjects/deeplearning_git/DeepLearningZeroToAll-master/data-03-diabetes.csv',
#                 delimiter=',', dtype=np.float32)
# x_data = xy[:, 0:-1]
# y_data = xy[:, [-1]]

batch_size = 10

filename_queue = tf.train.string_input_producer(
    ['/Users/hyeonakim/PycharmProjects/deeplearning_git/DeepLearningZeroToAll-master/data-03-diabetes.csv'],
    shuffle=False, name='filename_queue'
)


# file queue 에서 읽어오기
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# 값의 형식을 정의하기
record_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)


# 배치를 이용해서 읽어오기
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)


# suffle batch
min_afer_dequeue = 10000
capacity = min_afer_dequeue + 3 * batch_size
train_x_batch, train_y_batch = tf.train.shuffle_batch([xy[0:-1], xy[-1:]], batch_size=10,
                                                      capacity=capacity, min_after_dequeue=min_afer_dequeue)


# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[batch_size, 8])
Y = tf.placeholder(tf.float32, shape=[batch_size, 1])
W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


# 2
# Hypothesis using sigmoid
# tf.div(1., 1. + tf.exp(tf.matmul(X, W) + b))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# 3
# Cost/Loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 4
# Accuracy computation
# True if hypothesis>0.5 else False : cast function
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# 5
# Launch graph

with tf.Session() as sess:
    # Initialize Tensorflow variables
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(10001):
        x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_batch, Y: y_batch})
        if step % 200 == 0:
            print(step, cost_val)


    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_batch, Y: y_batch})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\n Accuracy: ", a)

    coord.request_stop()
    coord.join(threads)



