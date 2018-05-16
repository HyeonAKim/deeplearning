import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 이미지 불러오기
mnist_images = input_data.read_data_sets("C:\\Users\\HyunA\\PycharmProjects\\deeplearning\\dataset\\mnist",one_hot=False)

# train 객체에 담긴 첫 10개 이미지 labe(실제값) 가져오리
pixels,real_values = mnist_images.train.next_batch(10)
print("list of values loaded..",real_values)

example_to_visualize =5
print("element N " + str(example_to_visualize+1)+"of the list plotted")

image_ = pixels[example_to_visualize,:]
print(image_)
image = np.reshape(image_,[28,28])
print(image)
plt.imshow(image)
plt.show(block = True)
