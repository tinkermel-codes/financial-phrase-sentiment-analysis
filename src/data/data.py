import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.util import ngrams


def load_txt(path, sep="@", names=None, encoding="latin-1"):
    return pd.read_csv(path, sep=sep, encoding=encoding, names=names, header=None if names else "infer")

def clean_text(text, stop_words=None, lemmatizer=None):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]

    if lemmatizer is not None:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    if stop_words is not None:
        tokens = [t for t in tokens if t not in stop_words]

    return tokens


def get_ngrams(tokens, n):
    return [" ".join(gram) for gram in ngrams(tokens, n)]


def top_words_for_label(df, label, stop_words, n, lemmatizer=None):
    texts = " ".join(df[df["label"] == label]["text"])
    clean_tokens = clean_text(texts, stop_words, lemmatizer)
    return pd.Series(clean_tokens).value_counts().head(n)

def top_ngrams_for_label(df, label, stop_words, ngram_size, top_n, lemmatizer=None):
    texts = df.loc[df["label"] == label, "text"]

    tokens = []
    for t in texts:
        tokens.extend(clean_text(t, stop_words, lemmatizer))

    if ngram_size == 1:
        items = tokens
    else:
        items = get_ngrams(tokens, ngram_size)

    counts = pd.Series(items).value_counts()
    rel_freq = counts / counts.sum()

    return rel_freq.head(top_n)

def clean_text_for_tfidf(text, stop_words=None, lemmatizer=None):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]

    if lemmatizer is not None:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    if stop_words is not None:
        tokens = [t for t in tokens if t not in stop_words]

    return " ".join(tokens)

