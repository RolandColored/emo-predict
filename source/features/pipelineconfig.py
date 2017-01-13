from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler

from features.glovevectorizer import GloveVectorizer
from features.sentencelength import SentenceLength
from features.textextractor import TextExtractor


class PipelineConfig:

    @staticmethod
    def text_bow_plain():
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', min_df=1, max_df=0.9))

    @staticmethod
    def text_bow():
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', min_df=1, max_df=0.9),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=1000))

    @staticmethod
    def text_and_title_and_message_bow():
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
    def text_bow_title_and_message_glove(lang):
        return make_pipeline(FeatureUnion([
            ('text', make_pipeline(TextExtractor(column='text'),
                                   CountVectorizer(strip_accents='ascii', min_df=1, max_df=0.9),
                                   TfidfTransformer(),
                                   TruncatedSVD(n_components=1000))),
            ('title', make_pipeline(TextExtractor(column='title'),
                                    GloveVectorizer(lang=lang))),
            ('message', make_pipeline(TextExtractor(column='message'),
                                      GloveVectorizer(lang=lang))),
        ]))

    @staticmethod
    def text_avg_setence_length():
        return make_pipeline(TextExtractor(column='text'),
                             SentenceLength(),
                             StandardScaler())

