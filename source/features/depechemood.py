import textacy
from sklearn.base import BaseEstimator


class DepecheMood(BaseEstimator):

    def __init__(self, lang):
        self.lang = lang

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [textacy.lexicon_methods.emotional_valence(textacy.Doc(row, lang=self.lang)) for row in X]
