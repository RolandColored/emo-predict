import spacy
from sklearn.base import BaseEstimator


class Lemmatizer(BaseEstimator):

    def __init__(self, lang):
        self.nlp = nlp = spacy.load(lang)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ret_list = []
        for row in X:
            doc = self.nlp(row)
            new_row = ' '.join([word.lemma_ for word in doc])
            ret_list.append(new_row)
        return ret_list
