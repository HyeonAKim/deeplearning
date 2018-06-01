from keras import preprocessing
from PIL import Image,ImageColor



# 크롤링 이미지 키워드 입력
keyword = input('insert keyword : ')

# 이미지 저장 경로 확인
DIR = 'C:\\Users\\HyunA\\PycharmProjects\\deeplearning\\dataset\\raw\\'+keyword

if not os.path.exists(DIR):
    os.mkdir(DIR)

# 이미지 크롤링하기
crawling_image(keyword,100, 100,DIR)

# 이미지 변환하기
change_image(DIR,keyword,'png',50 ,'L')

# 데이터 한곳에 모으기 : 검색조건에 맞춰서 하나의 폴더에 넣기
SEARCH_DIR = 'C:\\Users\\HyunA\\PycharmProjects\\deeplearning\\dataset\\raw\\'
COPY_DIR = 'C:\\Users\\HyunA\\PycharmProjects\\deeplearning\\dataset\\'
search_move_image(SEARCH_DIR , COPY_DIR,'chair')

# train , test 데이터 폴더 생성하기

data_dir = 'C:\\Users\\HyunA\\PycharmProjects\\deeplearning\\dataset\\chair'
len(os.listdir(data_dir))

keras.prep



