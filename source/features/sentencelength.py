import nltk
from sklearn.base import BaseEstimator


class SentenceLength(BaseEstimator):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [[self._get_avg_sent_length(row)] for row in X]

    def _get_avg_sent_length(self, row):
        sentences = nltk.sent_tokenize(row)
        return sum(len(nltk.tokenize.casual.casual_tokenize(sentence)) for sentence in sentences) / len(sentences)

