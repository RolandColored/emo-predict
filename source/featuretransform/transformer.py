import spacy
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from featuretransform.document import Document


class Transformer:

    labels = []
    title_vectors = []
    common_nouns = []
    corpus = []

    def __init__(self, lang):
        self.nlp = spacy.load(lang)

    def process_row(self, row):
        self.corpus.append(row['text'])
        doc = Document(self.nlp, row)
        self.labels.append(row['labels'])
        self.title_vectors.append(doc.get_title_vector())
        self.common_nouns.append({noun_count[0]: noun_count[1] for noun_count in doc.get_common_nouns()})

    def get_common_nouns_vectorized(self):
        vec = DictVectorizer()
        features = vec.fit_transform(self.common_nouns)
        print(vec.get_feature_names())
        return features.toarray()

    def get_all_vectorized(self):
        vectorizer = CountVectorizer(min_df=0.1, max_df=0.7)
        tfidf = TfidfTransformer()

        data = vectorizer.fit_transform(self.corpus)
        data = tfidf.fit_transform(data)
        return data.toarray()
