"""
- Gradient descent 에서 학습률은 어떻게 설정할까?
    learning rate 가 너무 크다면 : OverShooting
    learning rate 가 너무 작다면 : takes too long, stops at local minimum

- Data preprocessing for gradient descent
    왜곡된 형태의 등고선이 만들어짐 > 조금만 값이 달라져도 크게 달라질 수 있음
    zero-centered data / normalized data
    standardization  = x - 평균 /  표준편차
    X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()

- Overfitting : 학습데이터에만 너무 잘 맞는 모델을  만들어 내는 경우
    More training data
    Reduce the number of features
    Regularization : 일반화 - Let's not have too big numbers in the weight.
    cost 함수에 reqularization term 을 추가함
    l2reg = 0.001 * tf.reduce_sum(tf.squre(W))

- Train & Test data set
- Performance evaluation : is this good?
    - train set으로 모델을 평가한다면 이미 다 알고 있기 때문에 100% 다 맞을 것
    - 30% 정도로 데이터로 분할해서 70%로 데이터를 학습하고 30% 데이터를 활용해서 모델을 평가
    - train , validation , testing 으로 나누기도 함. validation으로 하이퍼파라미터를 튜닝함

- 데이터 셋이 굉장히 많을 경우에는? online learning
    - train : 100 만개 일 경우 한번에 넣지 못함 > 10만개씩 잘라서 학습

- Accuracy
    - 실제 데이터의 Y 값과 모델예측 값을 비교해서 이미지 인식의 정확도는 95 % ~ 99% 정도
"""
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# 1
# Input Data
x_data = [[1, 2, 1], [1, 3, 2], [1, 3, 4], [1, 5, 5],
          [1, 7, 5], [1, 2, 5], [1, 6, 6], [1, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0],
          [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]

# Non-normalized inputs
# xy = np.array([[828.6599, 833.4500, 908100, 828.3499, 831.659973],
#                [823.0200, 828.0700, 1828100, 821.655029, 828.0700],
#                [819.9299, 824.2000, 1438100, 818.97998, 924.1499],
#                [816, 820.9589, 1008100, 815.4899, 819.2399]])
# xy = MinMaxScaler(xy)


# Evaluation our model using this test dataset.
x_test = [[2, 1, 1], [3, 1, 2], [3, 3, 4]]
y_test = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]

X = tf.placeholder(tf.float32, [None, 3])
Y = tf.placeholder(tf.float32, [None, 3])
W = tf.Variable(tf.random_normal([3, 3]))
b = tf.Variable(tf.random_normal([3]))

# 2
# Create hypothesis
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# Learning rate : NaN!
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.5).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-10).minimize(cost)


# 3
# Correct prediction Test model
prediction = tf.argmax(hypothesis, 1)
is_correct = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# 4
# Launch graph
with tf.Session() as sess:
    # Initailize Tensorflow variables
    sess.run(tf.global_variables_initializer())
    for step in range(201):
        cost_val, W_val, _ = sess.run([cost, W, optimizer],
                                      feed_dict={X: x_data, Y: y_data})
        print(step, cost_val, W_val)

    # Predict
    print("Prediction:", sess.run(prediction, feed_dict={X: x_test, Y: y_test}))
    # Calculate the accuracy
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))

