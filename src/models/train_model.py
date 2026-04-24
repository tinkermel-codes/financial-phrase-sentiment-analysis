from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data.clean_data import CleaningTransformer


def train_model(model, X_train, y_train, stop_words=None, lemmatizer=None, max_features=3000, ngram_range=(1, 2)):
    pipeline = Pipeline([
        ("clean", CleaningTransformer(stop_words, lemmatizer)),
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
        ("clf", model)
    ])

    pipeline.fit(X_train, y_train)
    return pipeline