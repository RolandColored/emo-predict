from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler

from features.depechemood import DepecheMood
from features.glovevectorizer import GloveVectorizer
from features.nrcemolex import NRCEmoLex
from features.posdistribution import PosDistribution
from features.readability import Readability
from features.sentencelength import SentenceLength
from features.textextractor import TextExtractor


class PipelineConfig:
    @staticmethod
    def text_bow_plain(lang):
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', min_df=1, max_df=0.9))

    @staticmethod
    def text_bow_100(lang):
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', min_df=1, max_df=0.9),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=100))

    @staticmethod
    def text_bow_250(lang):
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', min_df=1, max_df=0.9),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=250))

    @staticmethod
    def text_bow_500(lang):
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', min_df=1, max_df=0.9),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=500))

    @staticmethod
    def text_bow_1000(lang):
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', min_df=1, max_df=0.9),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=1000))

    @staticmethod
    def text_bigram_bow(lang):
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', min_df=0.001, max_df=0.8, ngram_range=(2, 2)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=500))

    @staticmethod
    def text_unigram_bigram_bow(lang):
        return make_pipeline(FeatureUnion([
            ('unigram', make_pipeline(TextExtractor(column='text'),
                                      CountVectorizer(strip_accents='ascii', min_df=1, max_df=0.9),
                                      TfidfTransformer(),
                                      TruncatedSVD(n_components=400))),
            ('bigram', make_pipeline(TextExtractor(column='title'),
                                     CountVectorizer(strip_accents='ascii', min_df=1, max_df=0.9, ngram_range=(2, 2)),
                                     TfidfTransformer(),
                                     TruncatedSVD(n_components=300))),
        ]))

    @staticmethod
    def text_and_title_and_message_bow(lang):
        return make_pipeline(FeatureUnion([
            ('text', make_pipeline(TextExtractor(column='text'),
                                   CountVectorizer(strip_accents='ascii', min_df=1, max_df=0.9),
                                   TfidfTransformer(),
                                   TruncatedSVD(n_components=400))),
            ('title', make_pipeline(TextExtractor(column='title'),
                                    CountVectorizer(strip_accents='ascii', min_df=1, max_df=0.9),
                                    TfidfTransformer(),
                                    TruncatedSVD(n_components=300))),
            ('message', make_pipeline(TextExtractor(column='message'),
                                      CountVectorizer(strip_accents='ascii', min_df=1, max_df=0.9),
                                      TfidfTransformer(),
                                      TruncatedSVD(n_components=300))),
        ]))

    @staticmethod
    def title_and_message_glove(lang):
        return make_pipeline(FeatureUnion([
            ('title', make_pipeline(TextExtractor(column='title'),
                                    GloveVectorizer(lang=lang))),
            ('message', make_pipeline(TextExtractor(column='message'),
                                      GloveVectorizer(lang=lang))),
        ]))

    @staticmethod
    def text_bow_title_and_message_glove(lang):
        return make_pipeline(FeatureUnion([
            ('text', PipelineConfig.text_bow(lang)),
            ('title', make_pipeline(TextExtractor(column='title'),
                                    GloveVectorizer(lang=lang))),
            ('message', make_pipeline(TextExtractor(column='message'),
                                      GloveVectorizer(lang=lang))),
        ]))

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
                             Readability(),
                             DictVectorizer(sparse=False),
                             StandardScaler())

    @staticmethod
    def text_sent(lang):
        return make_pipeline(TextExtractor(column='text'),
                             SentenceLength(),
                             StandardScaler())

    @staticmethod
    def text_pos(lang):
        return make_pipeline(TextExtractor(column='text'),
                             PosDistribution(lang),
                             DictVectorizer(sparse=False),
                             StandardScaler())

    @staticmethod
    def text_multiple_en(lang):
        return make_pipeline(FeatureUnion([
            ('text_pos', PipelineConfig.text_pos(lang)),
            ('text_readability', PipelineConfig.text_readability(lang)),
            ('text_depechemood', PipelineConfig.text_depechemood(lang)),
            ('text_emolex', PipelineConfig.text_emolex(lang)),
            ('text_sent', PipelineConfig.text_sent(lang)),
            ('text_bow', make_pipeline(TextExtractor(column='text'),
                                       CountVectorizer(strip_accents='ascii', min_df=1, max_df=0.9),
                                       TfidfTransformer(),
                                       TruncatedSVD(n_components=500)))
        ]))

    @staticmethod
    def text_multiple(lang):
        return make_pipeline(FeatureUnion([
            ('text_pos', PipelineConfig.text_pos(lang)),
            ('text_emolex', PipelineConfig.text_emolex(lang)),
            ('text_sent', PipelineConfig.text_sent(lang)),
            ('text_bow', make_pipeline(TextExtractor(column='text'),
                                       CountVectorizer(strip_accents='ascii', min_df=1, max_df=0.9),
                                       TfidfTransformer(),
                                       TruncatedSVD(n_components=500)))
        ], transformer_weights={'text_bow': 0.5, 'text_emolex': 0.3, 'text_pos': 0.15, 'text_sent': 0.05}))
