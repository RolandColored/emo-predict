import spacy

from featuretransform.document import Document


class Transformer:

    labels = []
    title_vectors = []

    def __init__(self, lang):
        self.nlp = spacy.load(lang)

    def process_row(self, row):
        doc = Document(self.nlp, row)
        self.labels.append(row['labels'])
        self.title_vectors.append(doc.get_title_vector())
