import csv
from scrapy.crawler import CrawlerProcess
from spiders.bild import BildSpider

spider = BildSpider()

process = CrawlerProcess({
    'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',
    'FEED_URI': '../articles/' + spider.name + '.csv',
    'FEED_FORMAT': 'csv',
})

process.crawl(spider)
process.start()

