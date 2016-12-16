import spacy
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from featuretransform.document import Document


class Transformer:

    labels = []
    title_glove_vectors = []
    message_glove_vectors = []
    common_nouns = []
    corpus = []

    def __init__(self, lang):
        self.nlp = spacy.load(lang)

    def process_row(self, row):
        self.corpus.append(row['text'])
        doc = Document(self.nlp, row)
        self.labels.append(row['labels'])
        self.title_glove_vectors.append(doc.get_title_glove_vector())
        self.message_glove_vectors.append(doc.get_message_glove_vector())
        self.common_nouns.append({noun_count[0]: noun_count[1] for noun_count in doc.get_common_nouns()})

    def get_title_glove_vectors(self):
        return self.title_glove_vectors

    def get_message_glove_vectors(self):
        return self.message_glove_vectors

    def get_common_nouns_vectors(self):
        vec = DictVectorizer()
        features = vec.fit_transform(self.common_nouns)
        return features.toarray()

    def get_all_count_vectors(self):
        vectorizer = CountVectorizer(min_df=0.01, max_df=0.8, ngram_range=(1, 1))
        tfidf = TfidfTransformer()

        data = vectorizer.fit_transform(self.corpus)
        data = tfidf.fit_transform(data)
        return data.toarray()
