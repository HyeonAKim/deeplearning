"""
활용방안
- LeNet-5
- AlexNet : imagenet 227*227*3 , 96필터 11* 11 filter , pooling 3 * 3 , Normalization layer를 사용 ( 요즘 잘 새용하지 않음 )
    - Relu를 처음 개발해서 사용
    - Dropout 도 사용
    - 배치사이즈 128
    - 동일한 네트워크 7개를 만들어서 앙상블로 해결 > 3%로 정도의 오류를 낮춤
- GoogleNet : Inception module ( 2014년 우승 )
    - 옆쪽으로 길게 늘어남 , 다른 형태의 conv를 사용
- ResNet : 2015년 우승 3.6%의 오류
    - Alexnet : 8 layers
    - Vgg : 19 layer
    - Resnet : 152 layer 사용

- 텍스트를 컨볼루션네트워크로 사용해보자고 함
- 알파고도 convolution 을 사용 > 네이쳐에 나옴
"""
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 1
# Set hyperparameter.
batch_size = 128
test_size = 256

# 2
# Create init_weight function.
def init_weihgt(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# 3
# Create model function.
def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d( l2, w3, strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx

# 4
# Input data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)
teX = teX.reshape(-1, 28, 28, 1)

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

# 5
# Set weight shape.
w = init_weihgt([3, 3, 1, 32]) # 3*3*1 conv, 32 outputs
w2 = init_weihgt([3, 3, 32, 64]) # 3*3*32 conv, 64 outputs
w3 = init_weihgt([3, 3, 64, 128]) # 3*3*64 conv, 128 outputs
w4 = init_weihgt([128 * 4 * 4, 625]) # FC 128*4*4 input, 624 outputs
w_o = init_weihgt([625, 10]) # FC 625inputs, 10 outputs (labels)

# 6
# Call model
p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

# 7
# Set cost, optimizer , predict
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# 8
# Lauch the graph in as session
with tf.Session() as sess:
    # initialize all variables.
    tf.global_variables_initializer().run()

    for i in range(100):
        train_batch = zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size))
        for start, end in train_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})
        test_indices = np.arange(len(teX))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) == sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                                                                 p_keep_conv: 1.0,
                                                                                                 p_keep_hidden: 1.0})))


