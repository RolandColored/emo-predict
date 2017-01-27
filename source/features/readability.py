import textacy
from sklearn.base import BaseEstimator


class Readability(BaseEstimator):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [textacy.text_stats.readability_stats(textacy.Doc(row)) for row in X]

