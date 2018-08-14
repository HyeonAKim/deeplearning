from PIL import Image,ImageColor
from icrawler.builtin import BaiduImageCrawler, BingImageCrawler, GoogleImageCrawler
from keras import preprocessing
import os
import shutil

# 이미지 크롤링 함수 (키워드 ,저장이미지최대수,이미지최소사이즈,  저장경로)
def crawling_image(keyword, max , min_size, dir):
    google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=10, storage={'root_dir': dir})
    google_crawler.crawl(keyword=keyword, offset=0, max_num=max,min_size=(min_size,min_size), max_size=None)
    print('complete download images')

# 이미지 사이즈 및 확장자 변경 함수 (이미지 저장 경로 , 변경확장자, 변경사이즈, 색상("L","RGBA"))
def  change_image(dir, extension , chage_size, color):
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
                elif color == 'RGBA':
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

def search_move_image(dir, copydir, keyword):
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

    print('complete move images')

# 크롤링 이미지 키워드 입력
keyword = input('insert keyword : ')

# 이미지 저장 경로 확인
DIR = 'C:\\Users\\HyunA\\PycharmProjects\\deeplearning\\dataset\\raw\\'+keyword

if not os.path.exists(DIR):
    os.mkdir(DIR)

# 이미지 크롤링하기
crawling_image(keyword,100, 100,DIR)

# 이미지 변환하기
change_image(DIR,'png',50 ,'L')

# 데이터 한곳에 모으기 : 검색조건에 맞춰서 하나의 폴더에 넣기
SEARCH_DIR = 'C:\\Users\\HyunA\\PycharmProjects\\deeplearning\\dataset\\raw\\'
COPY_DIR = 'C:\\Users\\HyunA\\PycharmProjects\\deeplearning\\dataset\\'
search_move_image(SEARCH_DIR , COPY_DIR,'chair')

# train , test 데이터 폴더 생성하기

data_dir = 'C:\\Users\\HyunA\\PycharmProjects\\deeplearning\\dataset\\chair'
len(os.listdir(data_dir))

keras.prep



