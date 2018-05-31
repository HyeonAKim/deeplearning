import os
import shutil
import random


data_dir = 'C:\\Users\\HyunA\\PycharmProjects\\deeplearning\\dataset\\chair'
train_dir = 'C:\\Users\\HyunA\\PycharmProjects\\deeplearning\\dataset\\train'
test_dir =  'C:\\Users\\HyunA\\PycharmProjects\\deeplearning\\dataset\\test'

percen = 30

# 493
# 반올림해서 정수 반환
print(round(len(os.listdir(data_dir))/30))
print(len(os.listdir(data_dir)))

# 디렉토리내에 파일 섞어서 랜덤한 데이터 옮기기
list =os.listdir(data_dir)
random.shuffle(list)

for filename in list[:round(len(os.listdir(data_dir))/30)]:
    shutil.copy(data_dir, train_dir)

for filename in list[round(len(os.listdir(data_dir))/30):]:
    shutil.copy(data_dir, test_dir)
