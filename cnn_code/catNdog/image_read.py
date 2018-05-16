import numpy as np
import matplotlib.pyplot as plt
from cnn_code.catNdog import read_data_set

# 이미지 불러오기
images = read_data_set.read_data_sets(train_dir="C:\\Users\\HyunA\\PycharmProjects\\deeplearning\\cnn_code", one_hot=False)

# train 객체에 담긴 첫 10개 이미지 labe(실제값) 가져오리
pixels,real_values = images.train.next_batch(10)
print("list of values loaded..",real_values)

example_to_visualize =5
print("element N " + str(example_to_visualize+1)+"of the list plotted")

image_ = pixels[example_to_visualize,:]
print(image_)
image = np.reshape(image_,[200,200])
print(image)
plt.imshow(image)
plt.show(block = True)

