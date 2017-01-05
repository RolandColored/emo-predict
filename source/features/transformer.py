from sklearn.pipeline import FeatureUnion


class Transformer:
    rows = []

    def __init__(self, lang, pipeline):
        self.lang = lang
        self.pipeline = pipeline

    def process_row(self, row):
        self.rows.append(row)

    def get_num_rows(self):
        return len(self.rows)

    def get_desc(self):
        steps = self.pipeline.steps
        feature_union = steps[0][1]
        if len(steps) == 1 and isinstance(feature_union, FeatureUnion):
            steps = feature_union.transformer_list

        return '\n'.join([str(estimator) for _, estimator in steps])

    def get_labels(self):
        return [row['labels'] for row in self.rows]

    def get_samples(self):
        return self.pipeline.fit_transform(self.rows)

