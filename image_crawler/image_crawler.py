from icrawler.builtin import BaiduImageCrawler, BingImageCrawler, GoogleImageCrawler

def getGoogleImage(keyword,dir,max):
    google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=10, storage={'root_dir': dir})
    google_crawler.crawl(keyword=keyword, offset=0, max_num=max,min_size=(100,100), max_size=None)
