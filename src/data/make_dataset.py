import pandas as pd


def load_txt(path, sep="@", names=None, encoding="latin-1"):
    return pd.read_csv(path, sep=sep, encoding=encoding, names=names, header=None if names else "infer")