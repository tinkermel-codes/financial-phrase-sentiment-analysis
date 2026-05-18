import pandas as pd
from nltk.util import ngrams
from src.data.clean_data import CleaningTransformer


def get_ngrams(tokens, n):
    return [" ".join(gram) for gram in ngrams(tokens, n)]


def top_words_for_label(df, label, stop_words, n, lemmatizer=None):
    cleaner = CleaningTransformer(stop_words, lemmatizer)
    texts = " ".join(df[df["label"] == label]["text"])
    clean_text = cleaner.transform(texts)
    clean_tokens = clean_text(texts, stop_words, lemmatizer)
    return pd.Series(clean_tokens).value_counts().head(n)


def top_ngrams_for_label(df, label, stop_words, ngram_size, top_n, lemmatizer=None):
    texts = df.loc[df["label"] == label, "text"]
    cleaner = CleaningTransformer(stop_words, lemmatizer)

    cleaned_texts = cleaner.transform(texts)
    tokens = " ".join(cleaned_texts).split()

    if ngram_size == 1:
        items = tokens
    else:
        items = get_ngrams(tokens, ngram_size)

    counts = pd.Series(items).value_counts()
    rel_freq = counts / counts.sum()

    return rel_freq.head(top_n)