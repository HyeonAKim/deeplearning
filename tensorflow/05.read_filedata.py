import numpy as np
import tensorflow as tf

# 1
# Input data set
xy = np.loadtxt('/Users/hyeonakim/PycharmProjects/deeplearning_git/data-01-test-score.txt', delimiter=',',
                dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data, len(y_data))


# 2
# place holder for a tensor that will be always fed.

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

# 3
# Cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 4
# Minimize Need a very small Learning rate for this data set.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# 5
# Launch the graph in a session.
sess = tf.Session()
# Initialize global variables in the graph.
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost:", cost_val, "\nPrediction:\n", hy_val)


# 6
# Ask my score
print("Your score will be ", sess.run(hypothesis,
                                      feed_dict={X: [[100, 70, 101]]}))




"""
# 대용량 데이터일 경우
# 1
# 읽을 파일 파악하기 
filename_queue = tf.train.string_input_producer(
    ['data-01-test-score.csv', 'data-02-test-score.csv',...],
    shuffle=False, name='filename_queue'
) 

# 2
# file queue 에서 읽어오기 
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# 3
# 값의 형식을 정의하기 
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

# 4
# 배치를 이용해서 읽어오기
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

# 5
# placeholder for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 6
# Hypothesis
hypothesis = tf.matmul(X, W) + b

# 7
# Simplified cost/loss function.
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e=5)
train = optimizer.minimize(cost)

# 8
# 세션 열기 
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 9
# Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train],
        feed_dict={X: x_batch, Y: y_batch})
    
    if step % 10 == 0 :
        print(step, "Cost: ", cost_val,
              "\nPrediction\n", hy_val)
        
coord.request_stop()
coord.join(threads)
"""



