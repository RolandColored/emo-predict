import math

import numpy as np
from sklearn.metrics import mean_squared_error


def root_mean_squared_error(y_true, y_pred,
                            sample_weight=None,
                            multioutput='uniform_average'):
    return math.sqrt(mean_squared_error(y_true, y_pred, sample_weight, multioutput))


def error_logger(y_true, y_pred,
                 sample_weight=None,
                 multioutput='uniform_average',
                 results_sink=None):
    if results_sink is not None:
        results_sink.write(np.sqrt((y_true - y_pred) ** 2))

    return root_mean_squared_error(y_true, y_pred, sample_weight, multioutput)
