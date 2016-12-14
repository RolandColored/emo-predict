from scrapy.crawler import CrawlerProcess

from spiders.foxnews import FoxnewsSpider

spider = FoxnewsSpider()

process = CrawlerProcess({
    'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',
    'FEED_URI': '../articles/' + spider.name + '.csv',
    'FEED_FORMAT': 'csv',
    'DOWNLOAD_DELAY': '0.25'
})

process.crawl(spider)
process.start()

