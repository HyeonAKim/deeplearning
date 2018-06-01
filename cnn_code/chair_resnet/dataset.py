import os
import shutil
import random
from PIL import Image,ImageColor
from icrawler.builtin import BaiduImageCrawler, BingImageCrawler, GoogleImageCrawler


# 이미지 크롤링 함수 (키워드 ,저장이미지최대수,이미지최소사이즈,  저장경로)
def crawling_image(keyword, max, min_size, dir):
    google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=10, storage={'root_dir': dir})
    google_crawler.crawl(keyword=keyword, offset=0, max_num=max, min_size=(min_size, min_size), max_size=None)
    print('complete download images')


# 이미지 사이즈 및 확장자 변경 함수 change_image()
# (이미지 저장 경로 , 변경확장자, 변경사이즈, 색상("L","RGBA"))
def  change_image(dir, keyword, extension , chage_size, color):
    #현재위치 파일 모두 가져온다.
    for filename in os.listdir(dir):

        # 파일 위치 경로
        img_dir = dir + '\\' + filename
        # slice_filename = filename[filename.find('.')+1:]
        save_dir = dir +'\\'+keyword+'_'+filename[:filename.find('.')+1]+extension
        # print(img_dir)
        # print(slice_filename)
        if filename[filename.find('.')+1:] == 'gif' :
            os.remove(img_dir)
        else :
            try :
                # 이미지 열기
                img = Image.open(img_dir)
                # 이미지 사이즈 조정
                resize_img = img.resize((chage_size, chage_size))
                # 이미지 컬러 변경
                if color == 'L':
                    color_img = resize_img.convert("L")
                elif color == 'RGB':
                    color_img = resize_img.convert("RGB")
                else :
                    color_img = resize_img
                # 기존 파일 삭제
                # os.remove(img_dir)
                # 이미지 저장 다른 이름
                color_img.save(save_dir, format=extension, compress_level=1)
            except :
                os.remove(save_dir)
                print('cannot open')

    print('complete change images')

# 데이터 폴더 merge 함수 : image_folder_merge()
# dir : 의자 > 의자. 의자들. 부서진 의자 등이 있는 폴더
# copydir : 병합할 폴더 경로
# keyword : 병합할 폴더명

def image_folder_merge(dir, copydir, keyword):
    copydir = copydir+keyword
    if not os.path.exists(copydir):
        os.mkdir(copydir)

    for dirname in os.listdir(dir):
        if dirname.find(keyword) is not -1 :
            print(dirname)
            for filename in os.listdir(dir+dirname):
                if filename.find(keyword) is not -1:
                    filedir = dir+dirname+'\\'+filename
                    shutil.copy(filedir,copydir)

    print('complete merge images')

# 모델학습을 위한 이미지 폴더 생성 : create_train_folder()
# image_dir : 저장된 이미지 경로
# save_dir : train, test 폴더가 저장될 경로
# train_percent : train 데이터 비율

def create_train_folder(image_dir, save_dir , train_percent):
    train_dir = save_dir+'\\train'
    test_dir = save_dir+'\\test'
    train_percent = float(train_percent)

    # 디렉토리내에 파일 섞어서 랜덤한 데이터 옮기기
    list = os.listdir(image_dir)
    random.shuffle(list)

    for filename in list[:round(len(os.listdir(image_dir)) / train_percent)]:
        shutil.copy(image_dir, train_dir)

    for filename in list[round(len(os.listdir(image_dir)) / train_percent):]:
        shutil.copy(image_dir, test_dir)
