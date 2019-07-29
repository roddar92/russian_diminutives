from pathlib import Path

import pandas as pd


def read_samples(path_to_corpus, columns):
    with Path(path_to_corpus).open() as fin:
        names = (line.split() for line in fin.readlines())
    return pd.DataFrame(names, columns=columns)