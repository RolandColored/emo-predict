from collections import Counter

import spacy
from sklearn.base import BaseEstimator


class PosDistribution(BaseEstimator):

    def __init__(self, lang):
        self.nlp = spacy.load(lang, parser=False)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._get_pos_distribution(row) for row in X]

    def _get_pos_distribution(self, row):
        doc = self.nlp(row)
        tags = [word.pos for word in doc]
        return {k: v / len(doc) for k, v in Counter(tags).items()}
