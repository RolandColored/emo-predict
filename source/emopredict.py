from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
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
from features.pipelineconfig import PipelineConfig
from features.transformer import Transformer
from resultwriter import ResultWriter

# data
result_writer = ResultWriter()
data_source = DataSource(['foxnews'])
print(data_source.get_num_rows(), "Samples processed")
transformer = Transformer(data_source, PipelineConfig.text_bow())
print('Datasource', data_source.get_desc())


# feature generation
print(transformer.get_desc())
samples = transformer.get_samples()
labels = transformer.get_labels()

num_features = len(samples[0])
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


def get_regressor_name(regressor_obj):
    if isinstance(regressor_obj, MultiOutputRegressor):
        regressor_name = regressor_obj.estimator.__class__.__name__
        if isinstance(regressor_obj.estimator, NuSVR):
            regressor_name += regressor_obj.estimator.kernel
    else:
        regressor_name = regressor_obj.__class__.__name__
    return regressor_name


# evaluate
for regressor in regressor_list:
    scores = cross_val_score(regressor, samples, labels)
    regressor_name = get_regressor_name(regressor)

    result_writer.add_result(regressor_name, scores.mean(), scores.std())
    print(regressor_name, 'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std()))

result_writer.write()
