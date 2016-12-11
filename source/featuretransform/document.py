import re
from collections import Counter

from sklearn.feature_extraction.text import strip_accents_ascii


class Document:

    def __init__(self, nlp, row):
        self.title = nlp(self._clean_text(row['title']))
        self.text = nlp(self._clean_text(row['text']))

    def get_title_vector(self):
        return self.title.vector

    def get_common_nouns(self):
        nouns_text = [token.lemma_ for token in self.text if token.pos_ in ['NOUN', 'PROPN']]
        nouns_title = [token.lemma_ for token in self.title if token.pos_ in ['NOUN', 'PROPN']]
        common_nouns = Counter(nouns_title) + Counter(nouns_text)
        return common_nouns.most_common(5)

    def _clean_text(self, text):
        text = strip_accents_ascii(text)
        text = text.replace('"', '')
        text = re.sub(r'(\([0-9]*\))', '', text)
        return text
