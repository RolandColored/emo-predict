import re

import spacy
from sklearn.feature_extraction.text import strip_accents_ascii


def _clean_text(text):
    text = strip_accents_ascii(text)
    text = text.replace('"', '')
    text = re.sub(r'(\([0-9]*\))', '', text)
    return text


class Transformer:
    desc = '?'
    rows = []
    nlp = None

    def __init__(self, lang, pipeline):
        self.lang = lang
        self.pipeline = pipeline

    def process_row(self, row):
        self.rows.append(row)

    def get_num_rows(self):
        return len(self.rows)

    def get_desc(self):
        return '\n'.join([str(estimator) for _, estimator in self.pipeline.steps])

    def get_labels(self):
        return [row['labels'] for row in self.rows]

    def get_samples(self):
        documents = [_clean_text(row['text']) for row in self.rows]
        return self.pipeline.fit_transform(documents)

    def get_title_message_glove(self):
        self.desc = "title+message glove vectors"
        return [(list(self._get_glove_vector(row['title'])) + list(self._get_glove_vector(row['message']))) for row in self.rows]

    def _get_glove_vector(self, text):
        if self.nlp is None:
            self.nlp = spacy.load(self.lang)
        return self.nlp(text).vector
