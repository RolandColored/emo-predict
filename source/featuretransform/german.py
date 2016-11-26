

class FeatureTransformGerman:

    def transform(self, row):
        return [len(row['text'])]

