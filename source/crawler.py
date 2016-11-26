import csv
from scrapy.crawler import CrawlerProcess
from spiders.bild import BildSpider

process = CrawlerProcess({
    'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
})

spider = BildSpider()
process.crawl(spider)
process.start()

results = spider.results

if len(results) == 0:
    print("No Results!")
else:
    print(len(results), " Results")

    with open('../articles/' + spider.name + '.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())

        writer.writeheader()
        writer.writerows(results)
