"""
3D 데이터셋에서 PCA를 적용해서 2D에 두영하는 간단한 선형오토인코더 생성
핸즈온머신러닝- 524page

# 출력의 개수가 입력의 개수와 동일합니다.
# 단순한 PCA를 수행하기 위해서는 활성함수를 사용하지 않으며 ( 즉, 모든 뉴런이 선형입니다.),
# 비용함수는 MSE
"""
# 1
# Setting
from __future__ import division, print_function, unicode_literals
# common
import tensorflow as tf
import numpy as np
import pathlib
import os
import sys


# 2
# Set random seed
def reset_graph(seed=45):
    tf.reset_default_graph()
    tf.set_random_seed( seed )
    np.random.seed( seed )


# 3
# Setting matplotlib : label size and font
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

plt.rcParams['font.family'] = 'NanumBarunGothic'
plt.rcParams['axes.unicode_minus'] = False

# 4
# Setting saved image folder
PROJECT_ROOT_DIR = "C:\\Users\\HyunA\\PycharmProjects\\deeplearning\\tensorflow_tutorial"
CHAPTER_ID = "autoencoder"


def save_fig(fig_id, tight_layout=True):
    pathlib.Path( os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID) ).mkdir( parents=True, exist_ok=True )
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

# 5
# create plot images function : plot_image() and plot_multiple_images()


def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    # https://matplotlib.org/gallery/images_contours_and_fields/interpolation_methods.html
    # interpolation = 'none' work well when a big image is scaled down.
    # interpolation = 'nearest' work well when a small images isa scaled up.
    plt.axis("off")


def plot_multiple_images(images, n_rows, n_cols, pad = 2):
    images = images - images.min()
    w, h = images.shape[1:]
    image = np.zeros(((w + pad) * n_rows + pad), (h + pad) * n_cols + pad)
    for y in range(n_rows):
        for x in range(n_cols):
            image[(y * (h + pad) + pad):(y * (h + pad) + pad + h), (x * (w + pad) + pad):(x * (w + pad) + pad + w)] = images[y * n_cols + x]
    plt.imshow(image, cmap="Greys", interpolation="nearest")
    plt.axis("off")

# 1
# input data
import numpy.random as rnd

rnd.seed(4)
m = 200
w1, w2 = 0.1, 0.3
noise = 0.1

angles = rnd.rand(m) * 3 * np.pi / 2 - 0.5
data = np.empty((m, 3))
data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * rnd.randn(m) / 2
data[:, 1] = np.sin(angles) * 0.7 + noise * rnd.randn(m) / 2
data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * rnd.randn(m)

# data rescale.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(data[: 100])  # 평균과 분산을 저장해 놓음
X_test = scaler.transform(data[100:])  # 훈련의 평균과 분산을 불러와서 정규화를 진행함


# 2
# Set hyper-parameter
learning_rate = 0.01  # 학습률 설정
n_iterations = 1000  # 반복횟수 설정

# 3
# Create Model

reset_graph()

n_inputs = 3
n_hidden = 2
n_outputs = n_inputs

X = tf.placeholder( tf.float32, shape=[None, n_inputs] )
hidden = tf.layers.dense( X, n_hidden )
outputs = tf.layers.dense( hidden, n_outputs )

# 4
# Set Loss function : MSE
reconstruction_loss = tf.reduce_mean( tf.square( outputs - X ) )

# 5
# Set optimizer function : Adam
optimizer = tf.train.AdadeltaOptimizer( learning_rate )
training_op = optimizer.minimize( reconstruction_loss )

# 6
# Initialize variables
init = tf.global_variables_initializer()

# 7
# Train model
codings = hidden  # 코딩을 만드는 은닉층을 출력합니다.

with tf.Session() as sess:
    init.run()
    for iteration in range( n_iterations ):
        training_op.run( feed_dict={X: X_train} )  # 레이블이 없음
    codings_val = codings.eval( feed_dict={X: X_test} )


# 8
# plot graph
fig = plt.figure(figsize=(4, 3))
plt.plot(codings_val[:, 0], codings_val[:, 1], "b.")
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
save_fig("linear_autoencoder_pca_plot")
plt.show()