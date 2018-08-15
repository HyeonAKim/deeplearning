import tensorflow as tf

# 1
# Input data set
# 변수의 갯수가 많을 때는 이런 방법을 사용하지 않음.
# x1_data = [73., 93., 89., 96., 73.]
# x2_data = [80., 88., 91., 98., 66.]
# x3_data = [75., 93., 90., 100., 70.]
# y_data = [152., 185., 180., 196., 142.]

# Matrix 를 사용
x_data = [[73., 93., 89.], [93., 96., 73.],
          [80., 88., 91.], [96., 98., 66.], [75., 93., 90.]]

y_data = [[152.], [185.], [180.], [196.], [142.]]

# 2
# place holder for a tensor that will be always fed.
# x1 = tf.placeholder(tf.float32)
# x2 = tf.placeholder(tf.float32)
# x3 = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
#
# w1 = tf.Variable(tf.random_normal([1]), name='weight1')
# w2 = tf.Variable(tf.random_normal([1]), name='weight2')
# w3 = tf.Variable(tf.random_normal([1]), name='weight3')
# b = tf.Variable(tf.random_normal([1]), name='bias')
#
# hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

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


