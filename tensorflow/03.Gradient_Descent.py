import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

# W = tf.Variable(tf.random_normal([1]), name='weight')
W = tf.Variable(5.0)
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# out hypothesis for linear model X * W.
hypothesis = X * W

# cost/Loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize : Gradient Descent using derivative : W:= learning_rate * derivative.
learninn_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X) * 2
descent = W - learninn_rate * gradient
update = W.assign(descent)

# 텐서에서 자동으로 미분해줌
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# train = optimizer.minimize(cost)

# get gradients : 코스트에 미분값을 구함
gvs = optimizer.compute_gradients(cost)
# apply gradient : 그레디언트 적용
apply_gradients = optimizer.apply_gradients(gvs)


# Launch the graph in a session
sess = tf.Session()

# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# 텐서의 미분계산값 확인
# for step in range(21):
#     # sess.run(update, feed_dict={X: x_data, Y: y_data})
#     print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
#     sess.run(train, feed_dict={X: x_data, Y: y_data})

# 수식, tf 미분 gradient 비교
for step in range(21):
    print(step, sess.run([gradient, W, gvs], feed_dict={X: x_data, Y: y_data}))
    sess.run(apply_gradients, feed_dict={X: x_data, Y: y_data})
