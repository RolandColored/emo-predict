from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.linear_model import MultiTaskLasso
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import NuSVR
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

from datasource import DataSource

# data
data_source = DataSource('bild')
for row in data_source.next_line():
    print(row)
exit()

# config
regressor_list = [
    DecisionTreeRegressor(random_state=0),
    RandomForestRegressor(random_state=0),
    KNeighborsRegressor(),
    MultiTaskLasso(random_state=0),
    MultiTaskElasticNet(random_state=0),
    MultiOutputRegressor(SVR(kernel='rbf', C=1e3, gamma=0.1)),
    MultiOutputRegressor(SVR(kernel='poly', C=1e3, degree=2)),
    MultiOutputRegressor(SVR(kernel='linear', C=1e3)),
    MultiOutputRegressor(NuSVR(C=1.0, nu=0.1)),
    MultiOutputRegressor(GradientBoostingRegressor(random_state=0)),
    MultiOutputRegressor(BayesianRidge()),
]


# evaluate
for regressor in regressor_list:
    scores = cross_val_score(regressor, data_source.data, data_source.target)

    if isinstance(regressor, MultiOutputRegressor):
        regressor_name = regressor.estimator.__class__.__name__
    else:
        regressor_name = regressor.__class__.__name__

    print(regressor_name, "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), scores)

