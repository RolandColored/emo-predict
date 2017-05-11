from sklearn.model_selection import train_test_split


class Transformer:
    def __init__(self, data_source, pipeline, name):
        self.data_source = data_source
        self.pipeline = pipeline
        self.name = name

    def get_desc(self):
        steps = self.pipeline.steps
        if 'featureunion' in self.pipeline.named_steps:
            steps = self.pipeline.named_steps['featureunion'].transformer_list

        return '\n'.join([str(estimator) for _, estimator in steps])

    def get_name(self):
        return self.name

    def get_labels(self):
        return [row['labels'] for row in self.data_source.rows]

    def get_samples(self):
        return self.pipeline.fit_transform(self.data_source.rows)

    def get_train_test_split(self):
        return train_test_split(self.get_samples(), self.get_labels(), test_size=0.2, random_state=42)

    def debug_alphabet(self):
        print(self.pipeline.named_steps['countvectorizer'].get_feature_names())
