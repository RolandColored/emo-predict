import textacy
from sklearn.base import BaseEstimator


class DepecheMood(BaseEstimator):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [textacy.lexicon_methods.emotional_valence(textacy.Doc(row)) for row in X]
