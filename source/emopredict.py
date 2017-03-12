import sys

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

from config.pipelineconfig import PipelineConfig
from config.regressors import regressor_list, get_regressor_name, get_n_jobs
from features.transformer import Transformer
from utils.datasource import DataSource
from utils.metrics import root_mean_squared_error
from utils.resultwriter import ResultWriter

# parameters: emopredict.py datasource1+datasource2+... pipeline n_jobs
data_source_names = sys.argv[1].split('+')
pipeline_name = sys.argv[2]
n_jobs = int(sys.argv[3])

result_writer = ResultWriter()
data_source = DataSource(data_source_names)
print(data_source.get_num_rows(), "Samples processed")

pipeline = getattr(PipelineConfig, pipeline_name)(data_source.get_lang())
transformer = Transformer(data_source, pipeline, pipeline_name)
print('Datasource', data_source.get_desc())


# feature generation
print(transformer.get_desc())
samples = transformer.get_samples()
labels = transformer.get_labels()

num_features = len(samples[0])
print("Generated", num_features, "feature dimensions")
result_writer.set_meta_data(data_source, transformer, num_features)


# evaluate
for regressor in regressor_list:
    regressor_name = get_regressor_name(regressor)
    scores = cross_val_score(regressor, samples, labels, n_jobs=get_n_jobs(regressor_name, n_jobs),
                             cv=10, scoring=make_scorer(root_mean_squared_error))

    result_writer.add_result(regressor_name, scores.mean(), scores.std())
    print(regressor_name, 'Accuracy: %0.4f (+/- %0.4f)' % (scores.mean(), scores.std()))

result_writer.write()
