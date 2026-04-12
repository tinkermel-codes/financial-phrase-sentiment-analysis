import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.util import ngrams


def load_txt(path, sep="@", names=None, encoding="latin-1"):
    return pd.read_csv(path, sep=sep, encoding=encoding, names=names, header=None if names else "infer")

def clean_text(text, stop_words=None):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]

    if stop_words is not None:
        tokens = [t for t in tokens if t not in stop_words]

    return tokens

def get_ngrams(tokens, n):
    return [" ".join(gram) for gram in ngrams(tokens, n)]

def top_words_for_label(df, label, stop_words, n):
    texts = " ".join(df[df["label"] == label]["text"])
    clean_tokens = clean_text(texts, stop_words)
    return pd.Series(clean_tokens).value_counts().head(n)

def top_ngrams_for_label(df, label, stop_words, ngram_size, top_n):
    texts = " ".join(df[df["label"] == label]["text"])
    clean_tokens = clean_text(texts, stop_words)

    if ngram_size == 1:
        items = clean_tokens
    else:
        items = get_ngrams(clean_tokens, ngram_size)

    return pd.Series(items).value_counts().head(top_n)