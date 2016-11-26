import csv


class DataSource:

    def __init__(self, name):
        self.name = name
        # self.data, self.target = make_regression(n_samples=100, n_targets=3, random_state=1)

    def next_line(self):
        with open('../articles/' + self.name + '.csv') as csvFile:
            reader = csv.DictReader(csvFile)
            for row in reader:
                yield row
        return None

    
