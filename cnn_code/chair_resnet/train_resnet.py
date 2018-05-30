from image_crawler import image_crawler
from PIL import Image
import os

# 크롤링 이미지 키워드 입력
keyword = input()
DIR = 'C:\\Users\\HyunA\\PycharmProjects\\deeplearning\\dataset\\'+keyword

# 이미지 저장 경로 확인
if not os.path.exists(DIR):
    os.mkdir(DIR)

# 이미지 크롤링 시작 (키워드 , 저장경로, 저장이미지수)
image_crawler.getGoogleImage(keyword,DIR, 100)

# 이미지 확장자 일괄 변환
def rename_file(DIR):
    #현재위치 파일 모두 가져온다.
    for filename in os.listdir(DIR):
        # 파일 위치 경로
        img_dir = DIR + '\\' + filename
        # slice_filename = filename[:filename.find('.')]
        save_dir = DIR +'\\'+filename[:filename.find('.')]+'.png'
        # print(img_dir)
        # print(slice_filename)
        try :
            # 이미지 열기
            img = Image.open(img_dir)
            # 이미지 사이즈 조정
            resize_img = img.resize((50, 50))
            # 이미지 저장 다른 이름
            resize_img.save(save_dir,format='png', compress_level=1)
        except :
            print('cannot open')

rename_file(DIR)
