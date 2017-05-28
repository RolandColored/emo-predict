from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler

from features.depechemood import DepecheMood
from features.glovevectorizer import GloveVectorizer
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
    def text_depechemood(lang):
        return make_pipeline(TextExtractor(column='text'),
                             DepecheMood(lang=lang),
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
    def text_bow_and_text_glove(lang):
        return make_pipeline(FeatureUnion([
            ('text_bow', PipelineConfig.text_tfidf_bow_1000(lang)),
            ('text_glove', PipelineConfig.text_glove(lang)),
        ]))

    @staticmethod
    def text_bow_and_text_glove_and_title_bow(lang):
        return make_pipeline(FeatureUnion([
            ('text_bow', PipelineConfig.text_tfidf_bow_1000(lang)),
            ('text_glove', PipelineConfig.text_glove(lang)),
            ('title_bow', PipelineConfig.title_tfidf_bow_1000(lang)),
        ]))

    @staticmethod
    def text_bow_and_text_depechemood(lang):
        return make_pipeline(FeatureUnion([
            ('text_bow', PipelineConfig.text_tfidf_bow_1000(lang)),
            ('text_depechemood', PipelineConfig.text_depechemood(lang)),
        ]))

    @staticmethod
    def text_bow_and_text_glove_and_title_bow_and_text_depechemood(lang):
        return make_pipeline(FeatureUnion([
            ('text_bow', PipelineConfig.text_tfidf_bow_1000(lang)),
            ('text_glove', PipelineConfig.text_glove(lang)),
            ('title_bow', PipelineConfig.title_tfidf_bow_1000(lang)),
            ('text_depechemood', PipelineConfig.text_depechemood(lang)),
        ]))

    @staticmethod
    def all_bow(lang):
        return make_pipeline(FeatureUnion([
            ('text_bow', PipelineConfig.text_tfidf_bow_1000(lang)),
            ('title_bow', PipelineConfig.title_tfidf_bow_1000(lang)),
            ('message_bow', PipelineConfig.message_tfidf_bow_1000(lang)),
        ]))

    @staticmethod
    def all_glove(lang):
        return make_pipeline(FeatureUnion([
            ('text_glove', PipelineConfig.text_glove(lang)),
            ('title_glove', PipelineConfig.title_glove(lang)),
            ('message_glove', PipelineConfig.message_glove(lang)),
        ]))

    @staticmethod
    def all_bow_and_glove(lang):
        return make_pipeline(FeatureUnion([
            ('text_bow', PipelineConfig.text_tfidf_bow_1000(lang)),
            ('title_bow', PipelineConfig.title_tfidf_bow_1000(lang)),
            ('message_bow', PipelineConfig.message_tfidf_bow_1000(lang)),
            ('text_glove', PipelineConfig.text_glove(lang)),
            ('title_glove', PipelineConfig.title_glove(lang)),
            ('message_glove', PipelineConfig.message_glove(lang)),
        ]))

    @staticmethod
    def _stop_words(lang):
        if lang == 'de':
            return stopwords.words('german')
        return stopwords.words('english')
