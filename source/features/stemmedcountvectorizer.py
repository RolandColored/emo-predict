from nltk.corpus import stopwords
from nltk.stem.snowball import GermanStemmer, EnglishStemmer
from sklearn.feature_extraction.text import CountVectorizer


class StemmedCountVectorizer(CountVectorizer):
    def __init__(self, lang, strip_accents=None, ngram_range=(1, 1), max_df=1.0, min_df=1):
        self.lang = lang

        if lang == 'de':
            self.stemmer = GermanStemmer()
            stop_words = stopwords.words('german')
        else:
            self.stemmer = EnglishStemmer()
            stop_words = stopwords.words('english')

        super(self.__class__, self).__init__(stop_words=stop_words, strip_accents=strip_accents,
                                             ngram_range=ngram_range, max_df=max_df, min_df=min_df)

    def _stem_tokens(self, words):
        return [self.stemmer.stem(w) for w in words]

    def build_analyzer(self):
        preprocess = self.build_preprocessor()
        stop_words = self.get_stop_words()
        tokenize = self.build_tokenizer()

        return lambda doc: self._word_ngrams(self._stem_tokens(
            tokenize(preprocess(self.decode(doc)))), stop_words)
