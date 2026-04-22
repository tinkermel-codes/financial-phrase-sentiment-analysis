from nltk.tokenize import word_tokenize
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

def clean_text(text, stop_words=None, lemmatizer=None):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]

    if lemmatizer is not None:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    if stop_words is not None:
        tokens = [t for t in tokens if t not in stop_words]

    return tokens


def clean_text_for_tfidf(text, stop_words=None, lemmatizer=None):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]

    if lemmatizer is not None:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    if stop_words is not None:
        tokens = [t for t in tokens if t not in stop_words]

    return " ".join(tokens)


def cleaning_transformer(stop_words=None, lemmatizer=None):
    return FunctionTransformer(
        lambda texts: [clean_text_for_tfidf(t, stop_words, lemmatizer) for t in texts],
        validate=False
    )


def make_cleaning_pipeline(stop_words=None, lemmatizer=None):
    return Pipeline([
        ("clean", cleaning_transformer(stop_words, lemmatizer))
    ])