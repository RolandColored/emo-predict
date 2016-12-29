from collections import ChainMap

from crawler.spiders.base import BaseSpider


class TheGuardianSpider(BaseSpider):
    name = 'theguardian'
    base = 'https://www.theguardian.com'

    def parse(self, response):
        title = self._clean(response.css('article header h1::text').extract_first())
        paragraphs = self._clean(' '.join(response.css('article header div.content__standfirst > p').extract()))
        paragraphs += self._clean(' '.join(response.css('article div.content__article-body > p').extract()))
        crawled_data = {'title': title, 'text': paragraphs}
        yield dict(ChainMap(response.meta['data'], crawled_data))
