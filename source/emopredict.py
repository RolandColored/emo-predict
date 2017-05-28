import os
import sys

from sklearn.metrics import make_scorer

from config.pipelineconfig_combined import PipelineConfig
from config.regressors import regressors_dict
from features.transformer import Transformer
from utils.datasource import DataSource
from utils.metrics import root_mean_squared_error
from utils.resultwriter import ResultWriter

# parameters: emopredict.py datasource1+datasource2+... pipeline regressor outputdir
data_source_names = sys.argv[1].split('+')
pipeline_name = sys.argv[2]
regressor_name = sys.argv[3]
output_dir = sys.argv[4]

# setup
source_dir = os.path.dirname(__file__)
data_source = DataSource(source_dir + '/../articles/', data_source_names)
result_writer = ResultWriter(output_dir + '/' + data_source.get_lang() + '_' + pipeline_name + '_' + regressor_name + '.csv')
print(data_source.get_num_rows(), "Samples processed")

pipeline = getattr(PipelineConfig, pipeline_name)(data_source.get_lang())
transformer = Transformer(data_source, pipeline, pipeline_name)
print('Datasource', data_source.get_desc())


# feature generation
print(transformer.get_desc())
X_train, X_test, y_train, y_test = transformer.get_train_test_split()

num_features = len(X_train[0])
print("Generated", num_features, "feature dimensions")
result_writer.set_meta_data(data_source, transformer, regressor_name, num_features)


# evaluate
# scorer = make_scorer(error_logger, results_sink=RawDataWriter())
scorer = make_scorer(root_mean_squared_error)
regressor = regressors_dict[regressor_name]
regressor.fit(X_train, y_train)
score = root_mean_squared_error(y_test, regressor.predict(X_test))

result_writer.add_result(score)
print(regressor_name, 'Error: %0.4f' % score)

result_writer.write()
