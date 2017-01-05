import spacy
from sklearn.base import BaseEstimator


class GloveVectorizer(BaseEstimator):

    def __init__(self, lang):
        self.nlp = spacy.load(lang)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.nlp(text).vector for text in X]

