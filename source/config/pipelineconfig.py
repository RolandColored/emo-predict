from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline

from features.lemmatizer import Lemmatizer
from features.stemmedcountvectorizer import StemmedCountVectorizer
from features.textextractor import TextExtractor


class PipelineConfig:
    @staticmethod
    def text_tfidf_bow_100(lang):
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=100))

    @staticmethod
    def text_tfidf_bow_250(lang):
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=250))

    @staticmethod
    def text_tfidf_bow_500(lang):
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=500))

    @staticmethod
    def text_tfidf_bow_1000(lang):
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=1000))



    @staticmethod
    def text_tfidf_stemmed_bow_100(lang):
        return make_pipeline(TextExtractor(column='text'),
                             StemmedCountVectorizer(lang, strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=100))

    @staticmethod
    def text_tfidf_stemmed_bow_250(lang):
        return make_pipeline(TextExtractor(column='text'),
                             StemmedCountVectorizer(lang, strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=250))

    @staticmethod
    def text_tfidf_stemmed_bow_500(lang):
        return make_pipeline(TextExtractor(column='text'),
                             StemmedCountVectorizer(lang, strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=500))

    @staticmethod
    def text_tfidf_stemmed_bow_1000(lang):
        return make_pipeline(TextExtractor(column='text'),
                             StemmedCountVectorizer(lang, strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=1000))




    @staticmethod
    def text_tfidf_lemmatized_bow_100(lang):
        return make_pipeline(TextExtractor(column='text'),
                             Lemmatizer(lang),
                             CountVectorizer(strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=100))

    @staticmethod
    def text_tfidf_lemmatized_bow_250(lang):
        return make_pipeline(TextExtractor(column='text'),
                             Lemmatizer(lang),
                             CountVectorizer(strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=250))

    @staticmethod
    def text_tfidf_lemmatized_bow_500(lang):
        return make_pipeline(TextExtractor(column='text'),
                             Lemmatizer(lang),
                             CountVectorizer(strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=500))

    @staticmethod
    def text_tfidf_lemmatized_bow_1000(lang):
        return make_pipeline(TextExtractor(column='text'),
                             Lemmatizer(lang),
                             CountVectorizer(strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=1000))




    @staticmethod
    def text_tfidf_bigram_bow_100(lang):
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', min_df=0.001, max_df=0.8, ngram_range=(2, 2),
                                             stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=100))

    @staticmethod
    def text_tfidf_bigram_bow_250(lang):
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', min_df=0.001, max_df=0.8, ngram_range=(2, 2),
                                             stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=250))

    @staticmethod
    def text_tfidf_bigram_bow_500(lang):
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', min_df=0.001, max_df=0.8, ngram_range=(2, 2),
                                             stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=500))

    @staticmethod
    def text_tfidf_bigram_bow_1000(lang):
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', min_df=0.001, max_df=0.8, ngram_range=(2, 2),
                                             stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=1000))




    @staticmethod
    def text_tfidf_bigram_stemmed_bow_100(lang):
        return make_pipeline(TextExtractor(column='text'),
                             StemmedCountVectorizer(lang, strip_accents='ascii', min_df=0.001, max_df=0.8,
                                                    ngram_range=(2, 2), stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=100))

    @staticmethod
    def text_tfidf_bigram_stemmed_bow_250(lang):
        return make_pipeline(TextExtractor(column='text'),
                             StemmedCountVectorizer(lang, strip_accents='ascii', min_df=0.001, max_df=0.8,
                                                    ngram_range=(2, 2), stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=250))

    @staticmethod
    def text_tfidf_bigram_stemmed_bow_500(lang):
        return make_pipeline(TextExtractor(column='text'),
                             StemmedCountVectorizer(lang, strip_accents='ascii', min_df=0.001, max_df=0.8,
                                                    ngram_range=(2, 2), stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=500))

    @staticmethod
    def text_tfidf_bigram_stemmed_bow_1000(lang):
        return make_pipeline(TextExtractor(column='text'),
                             StemmedCountVectorizer(lang, strip_accents='ascii', min_df=0.001, max_df=0.8,
                                                    ngram_range=(2, 2), stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=1000))



    @staticmethod
    def text_tfidf_bigram_lemmatized_bow_100(lang):
        return make_pipeline(TextExtractor(column='text'),
                             Lemmatizer(lang),
                             CountVectorizer(strip_accents='ascii', min_df=0.001, max_df=0.8, ngram_range=(2, 2),
                                             stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=100))

    @staticmethod
    def text_tfidf_bigram_lemmatized_bow_250(lang):
        return make_pipeline(TextExtractor(column='text'),
                             Lemmatizer(lang),
                             CountVectorizer(strip_accents='ascii', min_df=0.001, max_df=0.8, ngram_range=(2, 2),
                                             stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=250))

    @staticmethod
    def text_tfidf_bigram_lemmatized_bow_500(lang):
        return make_pipeline(TextExtractor(column='text'),
                             Lemmatizer(lang),
                             CountVectorizer(strip_accents='ascii', min_df=0.001, max_df=0.8, ngram_range=(2, 2),
                                             stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=500))

    @staticmethod
    def text_tfidf_bigram_lemmatized_bow_1000(lang):
        return make_pipeline(TextExtractor(column='text'),
                             Lemmatizer(lang),
                             CountVectorizer(strip_accents='ascii', min_df=0.001, max_df=0.8, ngram_range=(2, 2),
                                             stop_words=PipelineConfig._stop_words(lang)),
                             TfidfTransformer(),
                             TruncatedSVD(n_components=1000))


    @staticmethod
    def text_bow_100(lang):
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TruncatedSVD(n_components=100))

    @staticmethod
    def text_bow_250(lang):
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TruncatedSVD(n_components=250))

    @staticmethod
    def text_bow_500(lang):
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TruncatedSVD(n_components=500))

    @staticmethod
    def text_bow_1000(lang):
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TruncatedSVD(n_components=1000))



    @staticmethod
    def text_stemmed_bow_100(lang):
        return make_pipeline(TextExtractor(column='text'),
                             StemmedCountVectorizer(lang, strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TruncatedSVD(n_components=100))

    @staticmethod
    def text_stemmed_bow_250(lang):
        return make_pipeline(TextExtractor(column='text'),
                             StemmedCountVectorizer(lang, strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TruncatedSVD(n_components=250))

    @staticmethod
    def text_stemmed_bow_500(lang):
        return make_pipeline(TextExtractor(column='text'),
                             StemmedCountVectorizer(lang, strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TruncatedSVD(n_components=500))

    @staticmethod
    def text_stemmed_bow_1000(lang):
        return make_pipeline(TextExtractor(column='text'),
                             StemmedCountVectorizer(lang, strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TruncatedSVD(n_components=1000))




    @staticmethod
    def text_lemmatized_bow_100(lang):
        return make_pipeline(TextExtractor(column='text'),
                             Lemmatizer(lang),
                             CountVectorizer(strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TruncatedSVD(n_components=100))

    @staticmethod
    def text_lemmatized_bow_250(lang):
        return make_pipeline(TextExtractor(column='text'),
                             Lemmatizer(lang),
                             CountVectorizer(strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TruncatedSVD(n_components=250))

    @staticmethod
    def text_lemmatized_bow_500(lang):
        return make_pipeline(TextExtractor(column='text'),
                             Lemmatizer(lang),
                             CountVectorizer(strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TruncatedSVD(n_components=500))

    @staticmethod
    def text_lemmatized_bow_1000(lang):
        return make_pipeline(TextExtractor(column='text'),
                             Lemmatizer(lang),
                             CountVectorizer(strip_accents='ascii', stop_words=PipelineConfig._stop_words(lang)),
                             TruncatedSVD(n_components=1000))




    @staticmethod
    def text_bigram_bow_100(lang):
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', min_df=0.001, max_df=0.8, ngram_range=(2, 2),
                                             stop_words=PipelineConfig._stop_words(lang)),
                             TruncatedSVD(n_components=100))

    @staticmethod
    def text_bigram_bow_250(lang):
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', min_df=0.001, max_df=0.8, ngram_range=(2, 2),
                                             stop_words=PipelineConfig._stop_words(lang)),
                             TruncatedSVD(n_components=250))

    @staticmethod
    def text_bigram_bow_500(lang):
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', min_df=0.001, max_df=0.8, ngram_range=(2, 2),
                                             stop_words=PipelineConfig._stop_words(lang)),
                             TruncatedSVD(n_components=500))

    @staticmethod
    def text_bigram_bow_1000(lang):
        return make_pipeline(TextExtractor(column='text'),
                             CountVectorizer(strip_accents='ascii', min_df=0.001, max_df=0.8, ngram_range=(2, 2),
                                             stop_words=PipelineConfig._stop_words(lang)),
                             TruncatedSVD(n_components=1000))




    @staticmethod
    def text_bigram_stemmed_bow_100(lang):
        return make_pipeline(TextExtractor(column='text'),
                             StemmedCountVectorizer(lang, strip_accents='ascii', min_df=0.001, max_df=0.8,
                                                    ngram_range=(2, 2), stop_words=PipelineConfig._stop_words(lang)),
                             TruncatedSVD(n_components=100))

    @staticmethod
    def text_bigram_stemmed_bow_250(lang):
        return make_pipeline(TextExtractor(column='text'),
                             StemmedCountVectorizer(lang, strip_accents='ascii', min_df=0.001, max_df=0.8,
                                                    ngram_range=(2, 2), stop_words=PipelineConfig._stop_words(lang)),
                             TruncatedSVD(n_components=250))

    @staticmethod
    def text_bigram_stemmed_bow_500(lang):
        return make_pipeline(TextExtractor(column='text'),
                             StemmedCountVectorizer(lang, strip_accents='ascii', min_df=0.001, max_df=0.8,
                                                    ngram_range=(2, 2), stop_words=PipelineConfig._stop_words(lang)),
                             TruncatedSVD(n_components=500))

    @staticmethod
    def text_bigram_stemmed_bow_1000(lang):
        return make_pipeline(TextExtractor(column='text'),
                             StemmedCountVectorizer(lang, strip_accents='ascii', min_df=0.001, max_df=0.8,
                                                    ngram_range=(2, 2), stop_words=PipelineConfig._stop_words(lang)),
                             TruncatedSVD(n_components=1000))



    @staticmethod
    def text_bigram_lemmatized_bow_100(lang):
        return make_pipeline(TextExtractor(column='text'),
                             Lemmatizer(lang),
                             CountVectorizer(strip_accents='ascii', min_df=0.001, max_df=0.8, ngram_range=(2, 2),
                                             stop_words=PipelineConfig._stop_words(lang)),
                             TruncatedSVD(n_components=100))

    @staticmethod
    def text_bigram_lemmatized_bow_250(lang):
        return make_pipeline(TextExtractor(column='text'),
                             Lemmatizer(lang),
                             CountVectorizer(strip_accents='ascii', min_df=0.001, max_df=0.8, ngram_range=(2, 2),
                                             stop_words=PipelineConfig._stop_words(lang)),
                             TruncatedSVD(n_components=250))

    @staticmethod
    def text_bigram_lemmatized_bow_500(lang):
        return make_pipeline(TextExtractor(column='text'),
                             Lemmatizer(lang),
                             CountVectorizer(strip_accents='ascii', min_df=0.001, max_df=0.8, ngram_range=(2, 2),
                                             stop_words=PipelineConfig._stop_words(lang)),
                             TruncatedSVD(n_components=500))

    @staticmethod
    def text_bigram_lemmatized_bow_1000(lang):
        return make_pipeline(TextExtractor(column='text'),
                             Lemmatizer(lang),
                             CountVectorizer(strip_accents='ascii', min_df=0.001, max_df=0.8, ngram_range=(2, 2),
                                             stop_words=PipelineConfig._stop_words(lang)),
                             TruncatedSVD(n_components=1000))



    @staticmethod
    def _stop_words(lang):
        if lang == 'de':
            return stopwords.words('german')
        return stopwords.words('english')
