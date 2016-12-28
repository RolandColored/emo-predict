from collections import ChainMap

import scrapy
import csv
from w3lib.html import remove_tags, replace_escape_chars

from spiders.base import BaseSpider


class FoxnewsSpider(BaseSpider):
    name = 'foxnews'
    base = ['http://www.foxnews.com', 'http://insider.foxnews.com']

    def start_requests(self):
        with open('../fb-data/' + self.name + '.csv') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                if row['link'] is not None and (row['link'].startswith(self.base[0]) or row['link'].startswith(self.base[1])):
                    request = scrapy.Request(row['link'], self.parse)
                    request.meta['data'] = row
                    yield request

    def parse(self, response):
        if response.meta['data']['link'].startswith(self.base[0]):
            title = self._clean(response.css('article h1::text').extract_first())
            paragraphs = self._clean(' '.join(response.css('article div.article-text > p').extract()))
        else:
            title = self._clean(response.css('article h1::text').extract()[-1])
            paragraphs = self._clean(' '.join(response.css('article div.articleBody p').extract()))

        crawled_data = {'title': title, 'text': paragraphs}
        yield dict(ChainMap(response.meta['data'], crawled_data))
