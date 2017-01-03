from sklearn.base import BaseEstimator


class TextExtractor(BaseEstimator):

    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [row[self.column] for row in X]

