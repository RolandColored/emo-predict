from sklearn.base import BaseEstimator


class Printer(BaseEstimator):

    def __init__(self, nth=100):
        self.nth = nth

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(X[0::self.nth])
        return X

