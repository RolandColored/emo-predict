import csv

reaction_types = ['ANGRY', 'HAHA', 'LIKE', 'LOVE', 'SAD', 'WOW']


class DataSource:

    skip_counter = 0
    absolute_reactions = []

    def __init__(self, filenames):
        self.filenames = filenames

    def get_desc(self):
        return '+'.join(self.filenames)

    def next_row(self):
        for filename in self.filenames:
            with open('../articles/' + filename + '.csv') as csvFile:
                reader = csv.DictReader(csvFile)
                for row in reader:
                    reactions = [int(row[key]) for key in reaction_types]
                    reaction_count = sum(reactions)
                    self.absolute_reactions.append(reaction_count)

                    if reaction_count > 50:
                        row['labels'] = [value / reaction_count for value in reactions]
                        yield row
                    else:
                        self.skip_counter += 1
