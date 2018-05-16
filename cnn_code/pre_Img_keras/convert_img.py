# 이미지 전처리 방법  : https://blog.naver.com/cjsdyd2000/221226971713
# 1. 사진을 불러온다.
# 2. 사진의 사이즈를 획일화 한다.
# 3. 사진을 3차원 배열로 바꾼다 ( 세로길이, 가로길이, 깊이:색상)
# 4. 배열 데이터를 저장할 list를 선언하고 , 반복문을 사용해 list에 배열을 차곡차곡 쌓는다.
# 5. 반복문이 끝나면 학습데이터와 label이 만들어져있다.

import tensorflow as tf
import numpy as np

# 1.사진을 불러온다.
img = tf.keras.preprocessing.image.load_img('C:\\Users\\HyunA\\PycharmProjects\\deeplearning\\dataset\\dogNcat\\dogNcat_photos\\train-images\\0\\000001.jpg')
print(img.size)

# 2. 사진의 사이즈를 획일화 한다.
img_rs= img.resize((20,20))
print(img_rs.size)

# 3. 사진을 3차원 배열로 바꾼다.
array = tf.keras.preprocessing.image.img_to_array(img_rs)
print(array.shape)

# 4. 이미지 데이터를 쌓을 배열을 만들어 보자.

## list 배열 만들자.
List = np.array(array)

List = List[np.newaxis,:,:,:] # 차원을 늘리고, : 는 별도 작업을 하지 않는다는 의미
print(List.shape)
print(List)
print(List.size)

# 5. csv 파일로 떨궈서 확인해보자.
List = List.reshape(-1, 20*20*3)
List = List.astype('float32')
List /= 225
np.savetxt('image.txt',List,delimiter=',')
