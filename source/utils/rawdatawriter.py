import csv

from utils.datasource import reaction_types


class RawDataWriter:
    raw_data_csv = '../raw_data.csv'

    def __init__(self):
        with open(self.raw_data_csv, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(reaction_types)

    def write(self, data):
        with open(self.raw_data_csv, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)


