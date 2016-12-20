from collections import ChainMap

import scrapy
import csv
from w3lib.html import remove_tags, replace_escape_chars


class NytimesSpider(scrapy.Spider):
    name = 'nytimes'
    base = ['http://www.nytimes.com', 'https://www.nytimes.com']

    def start_requests(self):
        with open('../fb-data/' + self.name + '.csv') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                if row['link'] is not None and (row['link'].startswith(self.base[0]) or row['link'].startswith(self.base[1])):
                    request = scrapy.Request(row['link'], self.parse)
                    request.meta['data'] = row
                    yield request

    def parse(self, response):
        title = self._clean(response.css('article header h1::text').extract_first())
        paragraphs = self._clean(' '.join(response.css('article div.story-body > p').extract()))
        crawled_data = {'title': title, 'text': paragraphs}
        yield dict(ChainMap(response.meta['data'], crawled_data))

    def _clean(self, text):
        return replace_escape_chars(remove_tags(text)).replace('\xa0', ' ').strip()
