import re
from collections import Counter

import spacy
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, strip_accents_ascii


def _clean_text(text):
    text = strip_accents_ascii(text)
    text = text.replace('"', '')
    text = re.sub(r'(\([0-9]*\))', '', text)
    return text


class Transformer:

    lang = ''
    rows = []
    nlp = None

    def __init__(self, lang):
        self.lang = lang

    def process_row(self, row):
        self.rows.append(row)

        #self.corpus.append(row['text'])
        #doc = Document(self.nlp, row)
        #self.title_glove_vectors.append(doc.get_title_glove_vector())
        #self.message_glove_vectors.append(doc.get_message_glove_vector())
        #self.common_nouns.append({noun_count[0]: noun_count[1] for noun_count in doc.get_common_nouns()})

    def get_labels(self):
        return [row['labels'] for row in self.rows]

    def get_title_glove_vectors(self):
        return self.title_glove_vectors

    def get_message_glove_vectors(self):
        return self.message_glove_vectors

    def get_common_nouns_vectors(self):
        vec = DictVectorizer()
        features = vec.fit_transform(self.common_nouns)
        return features.toarray()

    def get_all_count_vectors(self):
        vectorizer = CountVectorizer(min_df=0.001, max_df=0.9, ngram_range=(1, 1))
        tfidf = TfidfTransformer()

        data = vectorizer.fit_transform([_clean_text(row['text']) for row in self.rows])
        data = tfidf.fit_transform(data)
        return data.toarray()

    def _init_nlp(self):
        if self.nlp is None:
            self.nlp = spacy.load(self.lang)
