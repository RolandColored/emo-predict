from collections import ChainMap

from crawler.spiders.base import BaseSpider


class NytimesSpider(BaseSpider):
    name = 'nytimes'
    base = ['http://www.nytimes.com', 'https://www.nytimes.com']

    def parse(self, response):
        title = self._clean(response.css('article header h1::text').extract_first())
        paragraphs = self._clean(' '.join(response.css('article div.story-body > p').extract()))
        crawled_data = {'title': title, 'text': paragraphs}
        yield dict(ChainMap(response.meta['data'], crawled_data))
