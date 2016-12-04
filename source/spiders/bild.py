from collections import ChainMap

import scrapy
import csv
from w3lib.html import remove_tags, replace_escape_chars


class BildSpider(scrapy.Spider):
    name = 'bild'
    base = 'http://www.bild.de'
    results = []

    def start_requests(self):
        with open('../fb-data/' + self.name + '.csv') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                if row['link'] is not None and row['link'].startswith(self.base):
                    request = scrapy.Request(row['link'], self.parse)
                    request.meta['data'] = row
                    yield request

    def parse(self, response):
        title = self._clean(response.css('article header h1::text').extract_first())
        paragraphs = self._clean(' '.join(response.css('article div.txt > p').extract()))
        crawled_data = {'title': title, 'text': paragraphs}
        #self.results.append(dict(response.meta['data'].items() | crawled_data.items()))
        yield dict(ChainMap(response.meta['data'], crawled_data))

    def _clean(self, text):
        return replace_escape_chars(remove_tags(text)).replace('\xa0', ' ').replace('„', '"').strip()
