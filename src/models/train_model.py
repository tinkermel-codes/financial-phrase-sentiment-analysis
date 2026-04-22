from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data.clean_data import cleaning_transformer


def build_tfidf_vectorizer(max_features, ngram_range):
    #return TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    return TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)


def build_pipeline(model, stop_words, lemmatizer, max_features, ngram_range):
    tfidf = build_tfidf_vectorizer(max_features, ngram_range)

    return Pipeline([
        ("clean", cleaning_transformer(stop_words, lemmatizer)),
        ("tfidf", tfidf),
        ("clf", model)
    ])


def train_model(model, X_train, y_train, stop_words, lemmatizer, max_features, ngram_range):
    pipeline = build_pipeline(model, stop_words, lemmatizer, max_features, ngram_range)
    pipeline.fit(X_train, y_train)
    return pipeline