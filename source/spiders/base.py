from collections import ChainMap

import scrapy
import csv
from w3lib.html import remove_tags, replace_escape_chars


class BaseSpider(scrapy.Spider):

    def start_requests(self):
        with open('../fb-data/' + self.name + '.csv') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                if row['link'] is not None and row['link'].startswith(self.base):
                    request = scrapy.Request(row['link'], self.parse)
                    request.meta['data'] = row
                    yield request

    def _clean(self, text):
        if text is not None:
            return replace_escape_chars(remove_tags(text)).strip()
        return ''
