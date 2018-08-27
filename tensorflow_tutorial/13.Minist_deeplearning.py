"""
activation funcation : 하나의 값이 다음 값으로 넘어갈때 임계치 이상이면 작동을 하고 아니면 작동을 하지 않도록 하는것

1. Xsigmoid 보다 Relu가 좋아
# 9단으로 연결을 해도 정확도가 높아지지 않음
# Backpropagation > 깊이가 깊은 레이어는 학습이 잘되지 않음 왜? > 시그모이드 게이트를 통과해서 미분값이 너무 작아지는 문제
# 1998 ~ 2006년도 까지 이문제를 해결하지 못함

2. Weight초기화를 잘해보자.
# 초기값을 굉장히 멍청하게 둠 > 왜? > 기존 : -1 ~ 1 사이의 랜덤값을 줌
# 초기값을 0 으로 준다면 ? > 미분*0 이되어서 미분값이 다 0 이 되버림
# 2006 : breaktrough > RBM을 사용 > Fine tunning
# RBM을 사용하지 않아도 괜찮음 : Xavier , He's initialization
# Glorot et al.2010
W = np.random.randn(fan_in, fan_out)/ np.sqrt(fan_in)
# He et al. 2015
W = np.random.randn(fan_in, fan_out)/np.sqrt(fan_in/2)
prettytensor implementaion - 사용

3. drop out
overfitting - train 데이터에 너무 학습된 상태 , 테스트 데이터에서 정확도가 낮게 나올경우
- overfitting 해결 방법
# train data 가 많아야함
# reqularization 을 하자
- 구부러진 함수를 피자
- dropout 하자
dropout_rate = tt.placeholder("float")
_L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
L1 = tf.nn.dropout(_L1, dropout_rate)

** 주의사항 : dropout은 학습할 때만 사용
Train
    sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, dropoutrate: 0.7}
Evaluation
    accuracy.eval({X: mnist.test.images, Y: mnist.test.labels, dropouttate: 1}

4. 앙상블
기계가 많고 학습 장비가 많을 경우  : 최소 2% ~ 4% 5% 까지 성능향상

5. 레고처럼 네트워크를 쌓아보자.
feedforward neural network.
Fast forward : resnet 구조
split and Merge : 입력을 나누워서 처리하고 하나로 모을 수 도 있음 ( convolutional network )
Recurrent network : RNN
The only limit is your imagination.

6. optimizer 의 종류
# simulation 페이지가 있음
# 처음의 시작은 adam 으로 하자 .


## summary
1. softmax vs Neural Nets for MNIST, 90 and 95.4%
2. Xavier initailization : 97.8%
3. Deep Neural Nets with Dropout : 98%
4. Adam and other optimizers
5. Exercise: Batch Normalization

"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 1
# Input Data
learning_rate = 0.001
training_epochs = 15
batch_size = 100

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# 2
# Create layers : with relu
# layer 5 : accuracy : 0.9764
# dropout : train > 0.5 ~ 0.7 / test > 1: accuracy : 0.9795

W1 = tf.get_variable("W1", shape=[784, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)


W3 = tf.get_variable("W3", shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([256]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

W4 = tf.get_variable("W4", shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([256]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)


W5 = tf.get_variable("W5", shape=[256, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L4, W5) + b5


# 3
# Define cost, optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 4
# Initialize : xavier initalization tensorflow_tutorial
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 5
# Train Model

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), "cost =", '{:.9f}'.format(avg_cost))

print('Learning Finished')

# 6
# Test Model
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict= {X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))
