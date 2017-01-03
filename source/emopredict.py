from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.linear_model import MultiTaskLasso
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import NuSVR
from sklearn.tree import DecisionTreeRegressor

from datasource import DataSource
from resultwriter import ResultWriter
from textextractor import TextExtractor
from transformer import Transformer

# data
result_writer = ResultWriter()
data_source = DataSource(['bild'])
transformer = Transformer('de', make_pipeline(
    TextExtractor(column='text'),
    CountVectorizer(min_df=1, max_df=0.9, strip_accents='ascii'),
    TfidfTransformer(),
    TruncatedSVD(n_components=1000)
))
print('Datasource', data_source.get_desc())


# feature generation
for row in data_source.next_row():
    transformer.process_row(row)
print(transformer.get_num_rows(), "Samples processed")

samples = transformer.get_samples()
labels = transformer.get_labels()

# debug vocubalery
# print(transformer.pipeline.steps[1][1].get_feature_names()); exit()

num_features = len(samples[0])
print(transformer.get_desc())
print("Generated", num_features, "features")
result_writer.set_meta_data(data_source, transformer, num_features)


# config
regressor_list = [
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    KNeighborsRegressor(),
    MultiTaskLasso(),
    MultiTaskElasticNet(),
    MultiOutputRegressor(BayesianRidge()),
    MultiOutputRegressor(GradientBoostingRegressor()),
    MultiOutputRegressor(NuSVR(kernel='rbf')),
    MultiOutputRegressor(NuSVR(kernel='poly')),
    MultiOutputRegressor(NuSVR(kernel='sigmoid')),
    ExtraTreesRegressor(),
    MultiOutputRegressor(AdaBoostRegressor()),
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

    result_writer.add_result(regressor_name, scores.mean(), scores.std())
    print(regressor_name, 'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std()))

result_writer.write()
