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
    desc = '?'
    rows = []
    nlp = None

    def __init__(self, lang):
        self.lang = lang

    def process_row(self, row):
        self.rows.append(row)

    def get_num_rows(self):
        return len(self.rows)

    def get_labels(self):
        return [row['labels'] for row in self.rows]

    def get_title_message_glove(self):
        self.desc = "title+message glove vectors"
        return [(list(self._get_glove_vector(row['title'])) + list(self._get_glove_vector(row['message']))) for row in self.rows]

    def get_bag_of_nouns_tfidf(self):
        common_nouns = []
        for row in self.rows:
            common_nouns.append({noun_count[0]: noun_count[1] for noun_count in self._get_common_nouns(row)})

        vectorizer = DictVectorizer()
        tfidf = TfidfTransformer()
        self.desc = str(vectorizer) + '\n' + str(tfidf)

        data = vectorizer.fit_transform(common_nouns)
        data = tfidf.fit_transform(data)
        return data.toarray()

    def get_bag_of_words_tfidf(self):
        vectorizer = CountVectorizer(min_df=0.002, max_df=0.9, ngram_range=(1, 1))
        tfidf = TfidfTransformer()
        self.desc = str(vectorizer) + '\n' + str(tfidf)

        data = vectorizer.fit_transform([_clean_text(row['text']) for row in self.rows])
        data = tfidf.fit_transform(data)
        return data.toarray()

    def _get_common_nouns(self, row):
        nouns_text = [token.lemma_ for token in self._get_nlp(row['text']) if token.pos_ in ['NOUN', 'PROPN']]
        nouns_title = [token.lemma_ for token in self._get_nlp(row['title']) if token.pos_ in ['NOUN', 'PROPN']]
        common_nouns = Counter(nouns_title) + Counter(nouns_text)
        return common_nouns.most_common(5)

    def _get_nlp(self, text):
        if self.nlp is None:
            self.nlp = spacy.load(self.lang)
        return self.nlp(text)
    
    def _get_glove_vector(self, text):
        return self._get_nlp(text).vector
