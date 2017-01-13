from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import FeatureUnion, make_pipeline

from features.glovevectorizer import GloveVectorizer
from features.textextractor import TextExtractor


class PipelineConfig:

    @staticmethod
    def text_bow():
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', min_df=1, max_df=0.9),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=1000))

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
