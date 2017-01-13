import csv

reaction_types = ['ANGRY', 'HAHA', 'LIKE', 'LOVE', 'SAD', 'WOW']


class DataSource:

    skip_counter = 0
    absolute_reactions = []
    rows = []

    def __init__(self, filenames):
        self.filenames = filenames
        self._read_files()

    def get_desc(self):
        return '+'.join(self.filenames)

    def get_lang(self):
        if any(source in self.filenames for source in ['bild', 'ihre.sz', 'spiegelonline']):
            return 'de'
        return 'en'

    def get_num_rows(self):
        return len(self.rows)

    def _read_files(self):
        for filename in self.filenames:
            with open('../articles/' + filename + '.csv') as csvFile:
                reader = csv.DictReader(csvFile)
                for row in reader:
                    self._read_row(row)

    def _read_row(self, row):
        reactions = [int(row[key]) for key in reaction_types]
        reaction_count = sum(reactions)
        self.absolute_reactions.append(reaction_count)

        if reaction_count <= 50:
            self.skip_counter += 1
        elif row['text'] == '':
            self.skip_counter += 1
        else:
            row['labels'] = [value / reaction_count for value in reactions]
            self.rows.append(row)

