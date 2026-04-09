import pandas as pd
from nltk.tokenize import word_tokenize
import string


def load_txt(path, sep="@", names=None, encoding="latin-1"):
    return pd.read_csv(path, sep=sep, encoding=encoding, names=names, header=None if names else "infer")

def clean_text(text, stop_words, lemmatizer):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    filtered_tokens = [word for word in lemmatized if word.lower() not in stop_words]
    clean_tokens = [word for word in filtered_tokens if word not in string.punctuation]
    return clean_tokens