# 이미지 처리를 위한 데이터를 모아둔다. : C:\Users\HyunA\PycharmProjects\deeplearning\dataset\dogNcat\dogNcat_photos
# 이미지 인식을 위한 라벨 txt 를 생성한다.  : C:\Users\HyunA\PycharmProjects\deeplearning\dataset\dogNcat\labels.txt
# 이미지 처리 참고 URL  :http://pythonstudy.xyz/python/article/406-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%B2%98%EB%A6%AC
import os
from PIL import Image
from array import *
from random import shuffle
import gzip
# 이미지 업로드 및 저장경로 설정
Names = [['C:\\Users\\HyunA\\PycharmProjects\\deeplearning\\dataset\\dogNcat\\dogNcat_photos\\train-images','train'], ['C:\\Users\\HyunA\\PycharmProjects\\deeplearning\\dataset\\dogNcat\\dogNcat_photos\\test-images','test']]

for name in Names:

    data_image = array('B')
    data_label = array('B')

    FileList =[]
    #폴더에 있는 파일명 읽어오기
    for dirname in os.listdir(name[0])[0:]:
        path = os.path.join(name[0],dirname)

        for filename in os.listdir(path):
            if filename.endswith(".jpg"):
                FileList.append(os.path.join(name[0],dirname,filename))

    shuffle(FileList)

    # 각각의 이미지와 라벨을 입력

    for filename in FileList:
        label = int(filename.split('\\')[9])

        Im = Image.open(open(filename,'rb'))
        # gray로 색상전환
        gray = Im.convert("L")
        resize = gray.resize((200,200))
        width, height = resize.size

        pixel = resize.load()
        # print(pixel[19,19])


        # 이미지의 픽셀값 데이터로 넣기
        for x in range(0,width):
            for y in range(0,height):
                data_image.append(pixel[y,x])

        data_label.append(label)

    hexval = "{0:#0{1}x}".format(len(FileList), 6)  # number of files in HEX

    # header for label array

    header = array('B')
    header.extend([0, 0, 8, 1, 0, 0])
    header.append(int('0x' + hexval[2:][:2], 16))
    header.append(int('0x' + hexval[2:][2:], 16))

    data_label = header + data_label

    # additional header for images array

    if max([width, height]) <= 256:
        header.extend([0, 0, 0, width, 0, 0, 0, height])
    else:
        raise ValueError('Image exceeds maximum size: 256x256 pixels');

    header[3] = 3  # Changing MSB for image data (0x00000803)

    data_image = header + data_image
    
    # 수정부분
    output_file = gzip.open(name[1] + '-images-idx3-ubyte.gz', 'wb')
    data_image.tofile(output_file)
    output_file.close()
    
    # 수정부분
    output_file = gzip.open(name[1] + '-labels-idx1-ubyte.gz', 'wb')
    data_label.tofile(output_file)
    output_file.close()

# gzip resulting files

#
# os.system('gzip train-images-idx3-ubyte')
# os.system('gzip train-labels-idx1-ubyte')
# os.system('gzip test-images-idx3-ubyte')
# os.system('gzip test-labels-idx1-ubyte')

