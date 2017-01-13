from sklearn.pipeline import FeatureUnion


class Transformer:

    def __init__(self, data_source, pipeline):
        self.data_source = data_source
        self.pipeline = pipeline

    def get_desc(self):
        steps = self.pipeline.steps
        feature_union = steps[0][1]
        if len(steps) == 1 and isinstance(feature_union, FeatureUnion):
            steps = feature_union.transformer_list

        return '\n'.join([str(estimator) for _, estimator in steps])

    def get_labels(self):
        return [row['labels'] for row in self.data_source.rows]

    def get_samples(self):
        return self.pipeline.fit_transform(self.data_source.rows)

    def debug_alphabet(self):
        print(self.pipeline.named_steps['countvectorizer'].get_feature_names())

