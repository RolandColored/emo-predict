from collections import ChainMap

import scrapy
import csv
from w3lib.html import remove_tags, replace_escape_chars

from spiders.base import BaseSpider


class SZSpider(BaseSpider):
    name = 'ihre.sz'
    base = 'http://www.sueddeutsche.de'

    def parse(self, response):
        title = self._clean(response.css('article section.header h2::text').extract()[-1])
        paragraphs = self._clean(' '.join(response.css('article section.body > ul > li').extract()))
        paragraphs += self._clean(' '.join(response.css('article section.body > p').extract()))
        crawled_data = {'title': title, 'text': paragraphs}
        yield dict(ChainMap(response.meta['data'], crawled_data))
