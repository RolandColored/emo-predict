
class Document:

    def __init__(self, nlp, row):
        self.title = nlp(row['title'])
        self.text = nlp(row['text'])

    def get_title_vector(self):
        return self.title.vector

    def get_common_nouns(self):
        None
