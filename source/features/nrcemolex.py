import os

import nltk
from py_lex import EmoLex
from sklearn.base import BaseEstimator


class NRCEmoLex(BaseEstimator):

    def __init__(self, lang):
        current_dir = os.path.dirname(__file__)
        self.lexicon = EmoLex(current_dir + '/../resources/NRC-emotion-lexicon-' + lang + '.txt')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.lexicon.summarize_doc(nltk.tokenize.casual.casual_tokenize(row)) for row in X]
