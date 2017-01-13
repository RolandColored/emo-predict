import re

from sklearn.base import BaseEstimator


class TextExtractor(BaseEstimator):

    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._pre_process_text(row[self.column]) for row in X]

    def _pre_process_text(self, text):
        # remove URLs
        text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
        return text

