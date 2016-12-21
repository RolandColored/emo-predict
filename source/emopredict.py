from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.linear_model import MultiTaskLasso
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import NuSVR
from sklearn.tree import DecisionTreeRegressor
from joblib import Parallel, delayed
from datasource import DataSource
from featuretransform.transformer import Transformer

# data
data_source = DataSource('foxnews')
transformer = Transformer('en')

print("Data: ", data_source.name)
print("Transformer lang: ", transformer.lang)

for row in data_source.next_row():
    transformer.process_row(row)

print("Data processed")

samples = transformer.get_title_message_glove_vectors()
labels = transformer.get_labels()

print("Generated ", len(samples[0]), " features")

# config
regressor_list = [
    DecisionTreeRegressor(random_state=0),
    RandomForestRegressor(random_state=0),
    KNeighborsRegressor(),
    MultiTaskLasso(random_state=0),
    MultiTaskElasticNet(random_state=0),
    MultiOutputRegressor(NuSVR(kernel='rbf')),
    MultiOutputRegressor(NuSVR(kernel='poly')),
    MultiOutputRegressor(NuSVR(kernel='sigmoid')),
    MultiOutputRegressor(GradientBoostingRegressor(random_state=0)),
    MultiOutputRegressor(BayesianRidge()),
]


# evaluate
def run_evaluation(regressor):
    scores = cross_val_score(regressor, samples, labels)

    if isinstance(regressor, MultiOutputRegressor):
        regressor_name = regressor.estimator.__class__.__name__
    else:
        regressor_name = regressor.__class__.__name__

    print(regressor_name, "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

Parallel(n_jobs=4)(delayed(run_evaluation)(regressor) for regressor in regressor_list)
