import sys

from sklearn.model_selection import cross_val_score

from config.pipelineconfig import PipelineConfig
from config.regressors import regressor_list, get_regressor_name
from datasource import DataSource
from features.transformer import Transformer
from resultwriter import ResultWriter

# parameters: emopredict.py datasource1+datasource2+... pipeline
result_writer = ResultWriter()
data_source = DataSource(sys.argv[1].split('+'))
print(data_source.get_num_rows(), "Samples processed")

pipeline = getattr(PipelineConfig, sys.argv[2])(data_source.get_lang())
transformer = Transformer(data_source, pipeline)
print('Datasource', data_source.get_desc())


# feature generation
print(transformer.get_desc())
samples = transformer.get_samples()
labels = transformer.get_labels()
print(samples)
num_features = len(samples[0])
print("Generated", num_features, "feature dimensions")
result_writer.set_meta_data(data_source, transformer, num_features)


# evaluate
for regressor in regressor_list:
    scores = cross_val_score(regressor, samples, labels)
    regressor_name = get_regressor_name(regressor)

    result_writer.add_result(regressor_name, scores.mean(), scores.std())
    print(regressor_name, 'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std()))

result_writer.write()
