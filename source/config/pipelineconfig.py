from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features.depechemood import DepecheMood
from features.glovevectorizer import GloveVectorizer
from features.nrcemolex import NRCEmoLex
from features.posdistribution import PosDistribution
from features.readability import Readability
from features.textextractor import TextExtractor


class PipelineConfig:

    @staticmethod
    def text_tfidf_bow_1000(lang):
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=1000))

    @staticmethod
    def text_glove(lang):
        return make_pipeline(TextExtractor(column='text'),
                             GloveVectorizer(lang=lang))

    @staticmethod
    def text_emolex(lang):
        return make_pipeline(TextExtractor(column='text'),
                             NRCEmoLex(lang),
                             DictVectorizer(sparse=False),
                             StandardScaler())

    @staticmethod
    def text_depechemood(lang):
        return make_pipeline(TextExtractor(column='text'),
                             DepecheMood(),
                             DictVectorizer(sparse=False),
                             StandardScaler())

    @staticmethod
    def text_readability(lang):
        return make_pipeline(TextExtractor(column='text'),
                             Readability(lang),
                             DictVectorizer(sparse=False),
                             StandardScaler())

    @staticmethod
    def text_pos(lang):
        return make_pipeline(TextExtractor(column='text'),
                             PosDistribution(lang),
                             DictVectorizer(sparse=False),
                             StandardScaler())



    @staticmethod
    def title_tfidf_bow_1000(lang):
        return make_pipeline(TextExtractor(column='title'),
                             CountVectorizer(strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=1000))

    @staticmethod
    def title_glove(lang):
        return make_pipeline(TextExtractor(column='title'),
                             GloveVectorizer(lang=lang))

    @staticmethod
    def title_emolex(lang):
        return make_pipeline(TextExtractor(column='title'),
                             NRCEmoLex(lang),
                             DictVectorizer(sparse=False),
                             StandardScaler())

    @staticmethod
    def title_depechemood(lang):
        return make_pipeline(TextExtractor(column='title'),
                             DepecheMood(),
                             DictVectorizer(sparse=False),
                             StandardScaler())

    @staticmethod
    def title_readability(lang):
        return make_pipeline(TextExtractor(column='title'),
                             Readability(lang),
                             DictVectorizer(sparse=False),
                             StandardScaler())

    @staticmethod
    def title_pos(lang):
        return make_pipeline(TextExtractor(column='title'),
                             PosDistribution(lang),
                             DictVectorizer(sparse=False),
                             StandardScaler())



    @staticmethod
    def message_tfidf_bow_1000(lang):
        return make_pipeline(TextExtractor(column='message'),
                             CountVectorizer(strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=1000))

    @staticmethod
    def message_glove(lang):
        return make_pipeline(TextExtractor(column='message'),
                             GloveVectorizer(lang=lang))

    @staticmethod
    def message_emolex(lang):
        return make_pipeline(TextExtractor(column='message'),
                             NRCEmoLex(lang),
                             DictVectorizer(sparse=False),
                             StandardScaler())

    @staticmethod
    def message_depechemood(lang):
        return make_pipeline(TextExtractor(column='message'),
                             DepecheMood(),
                             DictVectorizer(sparse=False),
                             StandardScaler())

    @staticmethod
    def message_readability(lang):
        return make_pipeline(TextExtractor(column='message'),
                             Readability(lang),
                             DictVectorizer(sparse=False),
                             StandardScaler())

    @staticmethod
    def message_pos(lang):
        return make_pipeline(TextExtractor(column='message'),
                             PosDistribution(lang),
                             DictVectorizer(sparse=False),
                             StandardScaler())


    @staticmethod
    def _stop_words(lang):
        if lang == 'de':
            return stopwords.words('german')
        return stopwords.words('english')
