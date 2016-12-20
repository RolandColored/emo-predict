from collections import ChainMap

import scrapy
import csv
from w3lib.html import remove_tags, replace_escape_chars


class SZSpider(scrapy.Spider):
    name = 'ihre.sz'
    base = 'http://www.sueddeutsche.de'

    def start_requests(self):
        with open('../fb-data/' + self.name + '.csv') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                if row['link'] is not None and row['link'].startswith(self.base):
                    request = scrapy.Request(row['link'], self.parse)
                    request.meta['data'] = row
                    yield request

    def parse(self, response):
        title = self._clean(response.css('article section.header h2::text').extract()[-1])
        paragraphs = self._clean(' '.join(response.css('article section.body > ul > li').extract()))
        paragraphs += self._clean(' '.join(response.css('article section.body > p').extract()))
        crawled_data = {'title': title, 'text': paragraphs}
        yield dict(ChainMap(response.meta['data'], crawled_data))

    def _clean(self, text):
        return replace_escape_chars(remove_tags(text)).strip()
