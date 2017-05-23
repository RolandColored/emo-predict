import os

import spacy
from iwnlp.iwnlp_wrapper import IWNLPWrapper
from sklearn.base import BaseEstimator


class Lemmatizer(BaseEstimator):

    def __init__(self, lang):
        self.lang = lang
        self.nlp = nlp = spacy.load(lang)
        current_dir = os.path.dirname(__file__)
        self.iwnlp = IWNLPWrapper(lemmatizer_path=current_dir + '/../resources/IWNLP.Lemmatizer_20170501.json')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ret_list = []
        for row in X:
            doc = self.nlp(row)

            # workaround until german lemmatizer is integrated in spacy
            if self.lang == 'de':
                new_row = ' '.join([self.iwnlp.lemmatize(str(word), word.pos_) for word in doc])
            else:
                new_row = ' '.join([word.lemma_ for word in doc])
            ret_list.append(new_row)
            print(new_row)
        return ret_list
