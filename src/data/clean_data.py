from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


class CleaningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, stop_words=None, lemmatizer=None):
        self.stop_words = stop_words
        self.lemmatizer = lemmatizer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self.clean_text)

    def clean_text(self, text):
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t.isalpha()]

        if self.lemmatizer is not None:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        if self.stop_words is not None:
            tokens = [t for t in tokens if t not in self.stop_words]

        return " ".join(tokens)