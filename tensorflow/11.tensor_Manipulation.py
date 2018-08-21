"""
- 레이어가 얼마 없으면 학습이 잘 되지 않더라.
- CIFAR : Canadian Institute for Advanced Research.
- Breakthrough
1. 굉장히 deep 한 레이어는 출력을 할 수 없다고 하였음.
2. 하지만 초기값을 잘주면 깊게 학습할 수 있다 : 2006년 > Deeplearnig 이라고 이름을 바꾸자.
3. 정말로 주목 받기 시작 : ImageNet Classification  대회에서 alex net 으로 오류가 갑자기 떨어짐 (15%) > 2015년 3% 로 떨어짐

- 왜 지금 우리가 이 분야를 공부해야할까?
1. Not too late to be a world expert.
2. Not too complicated.

1. 이전과는 달리 정확도가 높아짐
2. 텐서플로우와 같은 툴이 많이 공개되어 있음
3. 쉬운 파이썬 언어를 이용해서 프로그래밍할 수 있음
"""
import tensorflow as tf
import numpy as np

# 김밥과 같은 어레이 ㅋㅋㅋㅋ
t = np.array([0., 1., 2., 3., 4., 5., 6.])
print(t)
print(t.ndim) # rank
print(t.shape) # shape
print(t[0], t[1], t[-1])
print(t[:2], t[5:])

# 2D array
t = np.array([[1., 2., 3.], [4., 5., 6]])
print(t)
print(t.ndim) # rank
print(t.shape) # shape

# tensor
t = tf.constant([1, 2, 3, 4])
t = tf.constant([[1, 2],
                 [3, 4]])
tf.shape(t).eval()

# 괄호의 갯수 : rank  = 4 (?, ?, ?, ?)
t = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8],[9, 10, 11, 12]],
                  [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
tf.shape(t).eval()

t = tf.constant([
                    [
                        [
                            [1,2,3,4],
                            [5,6,7,8],
                            [9,10,11,12]
                        ],
                        [
                            [13,14,15,16],
                            [17,18,19,20],
                            [21,22,23,24]
                        ]
                    ]
                ])
# 축 : axis = 0 , 1, 2, 3(-1)

# Matmul VS multiply
matrix1 = tf.constant([[1., 2.], [3., 4.]]) # (?, ?) > (2, 2)
matrix2 = tf.constant([[1.], [2.]]) # (?, ?) > (2, 1)
print("Metrix 1 shape", matrix1.shape)
print("Metrix 2 shape", matrix2.shape)
tf.matmul(matrix1, matrix2).eval() # (2, 2) * (2, 1) = (2, 1)
# (matrix1*matrix2)

# Broadcasting : 차원이 맞지 않는데 차원을 맞춰주는 것이 Broadcasting 임.
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[1., 2.]])

# Reduce mean : 평균을 구하는데 줄여서 구함
tf.reduce_mean([1, 2], axis=0) # integer 일때 평균은  1이 나옴 반드시 float 형태로 변환

# axis = 0 , 1(-1):가장 안쪽
x = [[1., 2.],
     [3., 4.]]
tf.reduce_mean(x).eval() # 모든 값에 대한 평균
tf.reduce_mean(x, axis=0).eval() # 1,3의 평균
tf.reduce_mean(x, axis=1).eval() # 1,2의 평균

# Argmax : 축의 개념을 사용해서 쓸 수 있음 , 가장 큰값의 위치를 알려주는 것
x = [[0, 1, 2],
     [2, 1, 0]]
tf.argmax(x, axis=0).eval() # (0,2) , (1,1) , (2,0) 을 비교 > [1, 0, 0]
tf.argmax(x, axis=1).eval() # [2, 0]
tf.argmax(x, axis=-1).eval() # [2, 0]

# Reshape**
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]
              ])
# rank : 3 (?, ?, ?) > ( 2, 2, 3 )
tf.reshape(t, shape=[-1, 3]).eval()


# Reshape ( squeeze, expand )
tf.squeeze([[0], [1], [2]]).eval() # [0, 1, 2]
tf.expand_dims([0, 1, 2], 1).eval() # [[0], [1], [2]]

# One hot
t = tf.one_hot([[0], [1], [2], [0]], depth=3).eval() # shape 을 자동을 expand 함
tf.shape(t, shape=[-1, 3]).eval()

# Casting
tf.cast([1.8, 2.2, 3.3, 4.8], tf.int32).eval()
tf.cast([True, False, 1 == 1, 0 == 1], tf.int32).eval()

# Stack
x = [1, 4]
y = [2, 5]
z = [3, 6]

tf.stack([x, y, z]).eval()
tf.stack([x, y, z], axis=1).eval()

# Ones and Zeros like
x = [[0, 1, 2],
     [2, 1, 0]]
tf.ones_like(x).eval() # 똑같은 shape을 1로 채움
tf.zeros_like(x).eval() # 똑같은 shape를 0으로 채움

# Zip
for x, y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y)
