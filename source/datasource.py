import csv

reaction_types = ['ANGRY', 'HAHA', 'LIKE', 'LOVE', 'SAD', 'WOW']


class DataSource:

    def __init__(self, name):
        self.name = name

    def get_num_count(self):
        return 30000

    def next_row(self):
        with open('../articles/' + self.name + '.csv') as csvFile:
            reader = csv.DictReader(csvFile)
            for row in reader:
                reactions = [int(row[key]) for key in reaction_types]
                row['labels'] = reactions#[value / sum(reactions) for value in reactions]
                yield row

