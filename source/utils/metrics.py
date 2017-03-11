import math

from sklearn.metrics import mean_squared_error


def root_mean_squared_error(y_true, y_pred,
                       sample_weight=None,
                       multioutput='uniform_average'):
    return math.sqrt(mean_squared_error(y_true, y_pred, sample_weight, multioutput))

