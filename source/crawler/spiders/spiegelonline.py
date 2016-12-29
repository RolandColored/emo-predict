from collections import ChainMap

from crawler.spiders.base import BaseSpider


class SpiegelOnlineSpider(BaseSpider):
    name = 'spiegelonline'
    base = 'http://www.spiegel.de'

    def parse(self, response):
        title = self._clean(response.css('h2.article-title span.headline::text').extract_first())
        paragraphs = self._clean(' '.join(response.css('p.article-intro').extract()))
        paragraphs += self._clean(' '.join(response.css('div.article-section > p').extract()))
        crawled_data = {'title': title, 'text': paragraphs}
        yield dict(ChainMap(response.meta['data'], crawled_data))
