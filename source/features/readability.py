import textacy
from sklearn.base import BaseEstimator
from textacy.text_stats import TextStats


class Readability(BaseEstimator):

    def __init__(self, lang):
        self.lang = lang

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ret = []
        for row in X:
            stats = TextStats(textacy.Doc(row, lang=self.lang))
            stats_dict = stats.basic_counts

            if stats_dict['n_words'] > 0:
                if self.lang == 'de':
                    stats_dict['readability'] = stats.wiener_sachtextformel
                else:
                    stats_dict['readability'] = stats.gunning_fog_index
            else:
                stats_dict['readability'] = 0

            ret.append(stats_dict)
        return ret

