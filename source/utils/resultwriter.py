import csv
import time
from collections import OrderedDict
from statistics import mean, stdev


class ResultWriter:

    results = OrderedDict()

    def __init__(self, output_file):
        self.results['start_time'] = time.ctime()
        self.results['end_time'] = None
        self.output_file = output_file

    def set_meta_data(self, data_source, transformer, regressor_name, num_features):
        self.results['data_source'] = data_source.get_desc()
        self.results['regressor_name'] = regressor_name
        self.results['num_samples'] = data_source.get_num_rows()
        self.results['num_skipped'] = data_source.skip_counter
        self.results['reaction_count_mean'] = mean(data_source.absolute_reactions)
        self.results['reaction_count_stdev'] = stdev(data_source.absolute_reactions, self.results['reaction_count_mean'])
        self.results['feature_generator'] = transformer.get_name()
        self.results['num_features'] = num_features

    def add_result(self, score):
        self.results['error'] = score

    def write(self):
        self.results['end_time'] = time.ctime()

        with open(self.output_file, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.results.keys())

            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow(self.results)


