import re
from collections import Counter

import spacy
from numpy import concatenate
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
        # self.common_nouns.append({noun_count[0]: noun_count[1] for noun_count in doc.get_common_nouns()})

    def get_num_rows(self):
        return len(self.rows)

    def get_labels(self):
        return [row['labels'] for row in self.rows]

    def get_title_message_glove_vectors(self):
        print("title+message glove vectors")
        return [(list(self._get_glove_vector(row['title'])) + list(self._get_glove_vector(row['message']))) for row in self.rows]

    def get_common_nouns_vectors(self):
        vec = DictVectorizer()
        features = vec.fit_transform(self.common_nouns)
        return features.toarray()

    def get_all_count_vectors(self):
        vectorizer = CountVectorizer(min_df=0.001, max_df=0.9, ngram_range=(1, 1))
        print("CountVectorizer(min_df=", vectorizer.min_df, ", max_df=", vectorizer.max_df, ", ngram_range=",
              vectorizer.ngram_range, ")")
        tfidf = TfidfTransformer()

        data = vectorizer.fit_transform([_clean_text(row['text']) for row in self.rows])
        data = tfidf.fit_transform(data)
        return data.toarray()

    def _get_nlp(self) -> object:
        if self.nlp is None:
            self.nlp = spacy.load(self.lang)
        return self.nlp
    
    def _get_glove_vector(self, text):
        return self._get_nlp()(text).vector
