import csv

reaction_types = ['ANGRY', 'HAHA', 'LIKE', 'LOVE', 'SAD', 'WOW']


class DataSource:

    samples = {}
    targets = {}

    def __init__(self, name):
        self.name = name

        with open('../fb-data/' + self.name + '.csv') as csvFile:
            reader = csv.DictReader(csvFile, delimiter=';')
            for row in reader:
                reactions = [int(row[key]) for key in reaction_types]
                reactions_normalized = [value / sum(reactions) for value in reactions]
                self.targets[row['id']] = reactions_normalized

    def generate_features(self, transformer):
        with open('../articles/' + self.name + '.csv') as csvFile:
            reader = csv.DictReader(csvFile)
            for row in reader:
                features = transformer.transform(row)
                self.samples[row['id']] = features

    def get_data_set(self):
        X = []
        y = []
        for fb_id, sample in self.samples.items():
            X.append(sample)
            y.append(list(self.targets[fb_id]))
        return {'X': X, 'y': y}

