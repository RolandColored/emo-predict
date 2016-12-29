from collections import ChainMap

from w3lib.html import remove_tags, replace_escape_chars

from crawler.spiders.base import BaseSpider


class BildSpider(BaseSpider):
    name = 'bild'
    base = 'http://www.bild.de'

    def parse(self, response):
        title = self._clean(response.css('article header h1::text').extract_first())
        paragraphs = self._clean(' '.join(response.css('article div.txt > p').extract()))
        crawled_data = {'title': title, 'text': paragraphs}
        yield dict(ChainMap(response.meta['data'], crawled_data))

    def _clean(self, text):
        return replace_escape_chars(remove_tags(text)).replace('\xa0', ' ').replace('„', '"').strip()
