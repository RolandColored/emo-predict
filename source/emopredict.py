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
from featuretransform.transformer import Transformer

# data
data_source = DataSource('bild')
transformer = Transformer('de')

print("Initialized transformer")

for row in data_source.next_row():
    transformer.process_row(row)

print("Data processed")

samples = transformer.get_common_nouns_vectors()
labels = transformer.labels

print("Generated features")

# config
regressor_list = [
    DecisionTreeRegressor(random_state=0),
    RandomForestRegressor(random_state=0),
    KNeighborsRegressor(),
    MultiTaskLasso(random_state=0),
    MultiTaskElasticNet(random_state=0),
    MultiOutputRegressor(SVR(kernel='rbf')),
    MultiOutputRegressor(SVR(kernel='poly')),
    MultiOutputRegressor(NuSVR()),
    MultiOutputRegressor(GradientBoostingRegressor(random_state=0)),
    MultiOutputRegressor(BayesianRidge()),
]


# evaluate
for regressor in regressor_list:
    scores = cross_val_score(regressor, samples, labels)

    if isinstance(regressor, MultiOutputRegressor):
        regressor_name = regressor.estimator.__class__.__name__
    else:
        regressor_name = regressor.__class__.__name__

    print(regressor_name, "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

