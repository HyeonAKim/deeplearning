from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# imports
import  numpy as np
import  tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# convolution  개념
# convolution layers : 이미지에 특정 수의 convolution filter를 적용, 각 부분 마다 수학적 계산을 통해 출력 피쳐맵에 맞는 싱글 값을 계산.
# 컨볼루션 레이어에서 통상 적으로  relu  활성함수를 사용

# pooling layers : 컨볼루션 레이어에서 추출된  이미지 데이터를 다운샘플링한다. 통상적으로  max pooling을 사용함
# dense layers :  컨볼루션 레이어와 풀링레이어에 의해 다운샘플된 피쳐들을 분류하는 것을 수행
# 각  dense 레이어에서 모든 노드들은 앞의 레이어와 모두 연결되어 있다.

# 통상적으로  CNN은 특정을 추출하는 역할을 한다.

# CNN MNIST classifier
# 1.  convolutional layer #1 : 32개의 5*5 필터를 적용 , relu 활성함수
# 2. pooling layer #1 : 2*2 필터에 max pooling 적용, stride 2 (pooled 영역이 overlap 되지 않도록 함)
# 3. Convolutional layer #2 : 64개의 5*5 필터를 적용 , relu 활성함수
# 4. Pooling layer #2 : 2*2 필터에 max pooling 적용, stride 2
# 5. dense layer #1 : 1024 뉴런, 0.4 비율 드롭아웃을 통해 점칙화를함
# 6. dense layer #2 : 10개의 뉴런, 0-9의 클래스

# tf.layers : 3개의 레이어 타입을 만드는 메소드를 담은 모듈
# 1. conv2d() : 2차원 컨볼루션레이어 생성, 필터의 갯수 , 필터 커널 사이즈, padding, 활성함수 인자를 가짐
# 2. max_pooling2d() : 2차원 맥스 풀링레이어를 생성, 풀링 필터 사이즈, stride 인자를 가짐
# 3. dense() : dense 레이어 생성, 뉴런의 개수와 활성함수를 인자로 가짐

# cnn_model_fn  생성
def cnn_model_fn(features, labels, mode):
    """model function for cnn"""
    # input layer : 2차원의 이미지 데이터에서 컨볼루션레이어와 풀링레이어를 만들수있는 입력 텐서를 생성
    # argument :  [batch_size, image_height, image_width, channels] 기본값
    # batch_size : 학습동안 경사하강법을할 서브셋 사이즈
    # image_height : 이미지의 높이
    # image_width :  이미지의 넓이
    # channels : 이미지의 컬러채널 수 : 3 ( red, green, blue) , 1(black)
    # data_format  : 문자열로 입력과 인자 매칭

    # 1.이미지 입력

    input_layer = tf.reshape(features["x"],[-1,28,28,1])
    # 예제 이미지 28*28 , [batch_size, 28, 28,1] , feature 입력 피쳐 맵을 다음과 같이 shape 오퍼레이션을 이용해서 생성
    # batch_size = -1 : 입력 값의 수에 따라 동적으로 계산
    # features["x"] : 다른 모든 차원의 크기를 일정하게 유지

    # 2. convolution layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters = 32,
        kernel_size=[5,5],
        padding="same",
        activation= tf.nn.relu
    )
    # 입력레이어에 32 5*5필터와, relu 활성함수를 적용 : layer 모듈에 conv2d() 메소드 사용
    # input layer 는 텐서형태의 입력값이고 [ 배치사이즈, 이미지높이, 이미지넓이, 채널] 정보가 반드시 있어야함.
    # fileters: 적용할 필터의 개수
    # kernel_size : 차원 필터의 값 [높이, 넓이]
    # tip: 높이와 넓이의 값이 같다면 단일 값으로 설정해도 됩 . kernel_size = 5
    # padding : valid, same 이 있음 . 입력 텐서와 동일한 높이와 넓이를 가지는 출력텐서를 가진다면 same, 엣지에 0의 값을 더함
    # 패딩을 사용하지 않으면 28x28 텐서에서 5x5 컨볼 루션을 수행하면 24x24 텐터가 생성되므로 28x28 그리드에서 5x5 타일을 추출 할 수 있습니다.??
    # activation : tf.nn.relu에서 사요하는 relu 활성함수를 적용
    # con2d를 진행해서 나오는 결과텐서는 [batch_size, 28,28,32] : 같은 높이와 넓이 차원을 입력으로하되 32개의 채널을 가지고 있음


    # 3.pooling layer #2
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2],strides=2)
    # inputs : [batch_size, image_height, image_width, channels]을 가진 텐서 : 여기서 conv1이 입력이 됨
    # pool_size : 맥스풀링 필터를 사용 [높이, 넓이] , 높이와 넓이가 같다면 싱글 값으로 가능
    # strides : 보폭 사이즈 , 가로 , 세로 다른 보폭을 가지게 하려면 리스트 형태로 입력 가능 stride = [3,6]
    # 출력 : [batch_size, 14,14, 32]

    # 4.convloitionlayer #2 and pooling layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # conv2 출력 : [batch_size, 14,14, 64]
    # pool2 출력 : [batch_size , 7,7, 64 ]

    # 5.dense layer
    pool2_flat =  tf.reshape(pool2, [-1, 7*7*64])
    # batch_size : -1
    # feature 차원 : 3,136
    # pool2_flat [batch_size, 3136]

    # 6. dense() 메소드 레이어
    dense = tf.layers.dense(inputs=pool2_flat, units = 1024, activation=tf.nn.relu)
    # input : poolw_flat 된 형태의 텐서
    # units : 왜 1024개?
    # activation : relu

    #7. dropout reqularization
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN
    )
    # rate = 0.4 : 훈련동안 랜덤하게 40%의 elements drop out
    # training 모드 ; cnn_model_fn에 mode를 확인하고 전달
    # 출력 : [batch_size, 1024]

    #8. logits layer
    logits = tf.layers.dense(inputs=dropout, units=10)
    # 10개의 유닛 : 0-9 까지의 클래스
    # 활성함수 : 기본값 선형함수
    # 출력 : 10 차원의 텐서 [batch_size, 10]

    #9. generate predictions # predicted class/ probabilities

    #predicted class
    tf.argmax(input = logits, axis=1)
    # tf.argmax : logit 텐서와 highest raw value와 부합하는 class 예측을 할 수 있는 함수
    # input ; maximum valuse in logits
    # axis : 가장 큰 값을 찾을 입력 텐서의 축을 지정
    # 지수가 1인 차원에서 가장 큰 값을 찾고 싶음


    # probabilities
    tf.nn.softmax(logits, name="softmax_tensor")

    # Estimator Spec
    predictions = {
        "classes" : tf.argmax(input = logits, axis=1),
        "probabilities" : tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return  tf.estimator.EstimatorSpec(mode=mode, predictions= predictions)

    # 10. loss 계산
    # 얼마나 예측이 잘되는지 측정하기 위한 loass function이 필요 .
    # 다중 분류일 경우에는 통상적으로 cross entropy를 사용
    # cross entropy 은 train 과 eval 모드에서 모두 사용

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # labels : 크로스엔트로피 계산을 위해 labels 의 값들을 one-hot 인코딩
    # tf.one_hot :
        #   indices : 텐서에서 1 값의 위치.
        #   depth : 타켓 클라스의 갯수
    # onehot_labes
    # tf.losses.softmax_cross_entropy
        # argument : onehot_labels , logits : 8. 마지막 logits layer의 값
        # 출력 : loss , scalar tensor

    #11. 훈련 최적값 설정
    # 학습률 : 0.001 / 최적화 알고리즘 : stochastic gradient descent

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # 12. 평가매트릭스 추가
    eval_metric_ops = {
        "accuracy":tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return  tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# 훈련 및 평가

# 1. load training and test data
def main(unused_argv) :
    # 훈련 및 평가데이터 로드
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # 2. estimator 생성
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/mnist_convent_model")

    # model_fn : 훈련, 평가, 예측을 위해 사용할 함수
    # model_dir : 훈련을 한다음 저장된 모델을 이용할 때 사용

    # 3. logging hook
    # tf.train.LoggingTensorHook : 훈련을 하는동한 진행과정을 로깅할 수 있다.

    tensor_to_log = {"probabilities":"softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensor_to_log, every_n_iter=50)

    # 4. 모델 훈련하기
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    # x:  feature data  , y : labels
    # batch_size : 100
    # num_epochs : 지정된 단계 수에 도달 할 때까지 모델이 훈련 할 것임을 의미합니다.
    # shuffle : 훈련데이터를 섞을 것인지?
    # train : steps : 20000 2만번을 학습할 예정
    # logging_hook  : 훈련동안 로그를 남김

    # 5. 모델 평가하기
    # evaluate 메소드 사용 ,model_fn에 있는 eval_metric_ops 인자
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
  tf.app.run()