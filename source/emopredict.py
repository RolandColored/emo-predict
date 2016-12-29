import csv
import time
from collections import OrderedDict
from statistics import mean, stdev

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

from datasource import DataSource
from transformer import Transformer

# data
results = OrderedDict()
results['start_time'] = time.ctime()
results['end_time'] = None

data_source = DataSource(['bild', 'spiegelonline', 'ihre.sz'])
transformer = Transformer('de')
print('Datasource', data_source.get_desc())


# feature generation
for row in data_source.next_row():
    transformer.process_row(row)
print(transformer.get_num_rows(), "Samples processed")

samples = transformer.get_bag_of_words_tfidf()
labels = transformer.get_labels()
print(transformer.desc)


# result data
results['data_source'] = data_source.get_desc()
results['num_samples'] = transformer.get_num_rows()
results['num_skipped'] = data_source.skip_counter
results['reaction_count_mean'] = mean(data_source.absolute_reactions)
results['reaction_count_stdev'] = stdev(data_source.absolute_reactions, results['reaction_count_mean'])
results['feature_generator'] = transformer.desc
results['num_features'] = len(samples[0])
print("Generated", results['num_features'], "features")


# config
regressor_list = [
    DecisionTreeRegressor(random_state=0),
    RandomForestRegressor(random_state=0),
    KNeighborsRegressor(),
    MultiTaskLasso(random_state=0),
    MultiTaskElasticNet(random_state=0),
    MultiOutputRegressor(BayesianRidge()),
    MultiOutputRegressor(GradientBoostingRegressor(random_state=0)),
    MultiOutputRegressor(NuSVR(kernel='rbf')),
    MultiOutputRegressor(NuSVR(kernel='poly')),
    MultiOutputRegressor(NuSVR(kernel='sigmoid')),
]


# evaluate
for regressor in regressor_list:
    scores = cross_val_score(regressor, samples, labels, n_jobs=4)

    if isinstance(regressor, MultiOutputRegressor):
        regressor_name = regressor.estimator.__class__.__name__
        if isinstance(regressor.estimator, NuSVR):
            regressor_name += regressor.estimator.kernel
    else:
        regressor_name = regressor.__class__.__name__

    results[regressor_name + '_score'] = scores.mean()
    results[regressor_name + '_stdev'] = scores.std()
    print(regressor_name, 'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std()))

results['end_time'] = time.ctime()


# write results
with open('../results.csv', 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=results.keys())

    if csvfile.tell() == 0:
        writer.writeheader()
    writer.writerow(results)
