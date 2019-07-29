from collections import defaultdict
from pathlib import Path

import pandas as pd


class EthaloneCorporaCollector:
    def __init__(self):
        self.ethalone = defaultdict(list)

    @staticmethod
    def read_samples(path_to_corpus, columns):
        with Path(path_to_corpus).open() as fin:
            names = (line.split() for line in fin.readlines())
        return pd.DataFrame(names, columns=columns)

    def collect_corpora(self, path_to_corpus, path_to_output_corpus):
        for _, row in self.read_samples(path_to_corpus, ['name', 'diminutive']).iterrows():
            self.ethalone[row['name']].append(row['diminutive'])
            if row['name'].endswith('аша') or row['name'].endswith('уля'):
                base = row['name'][:-1] if row['name'].endswith('аша') else row['name'][:-3]
                for ending in ['уля', 'улечка', 'улик', 'уленька', 'утка', 'уточка']:
                    if base + ending not in self.ethalone[row['name']]:
                        self.ethalone[row['name']].append(base + ending)
            elif row['name'].lower().endswith('оля') or row['name'].endswith('уся') or row['name'].endswith('ася'):
                base = row['name'][:-1]
                for ending in ['ечка', 'ик', 'енька', 'юшка']:
                    if base + ending not in self.ethalone[row['name']]:
                        self.ethalone[row['name']].append(base + ending)
            elif row['name'].lower().endswith('иша'):
                base = row['name'][:-1]
                for ending in ['ечка', 'енька', 'уля', 'улечка', 'улик', 'уленька', 'утка', 'уточка']:
                    if base + ending not in self.ethalone[row['name']]:
                        self.ethalone[row['name']].append(base + ending)

        pd.DataFrame(list(self.ethalone.items())).to_csv(path_to_output_corpus)


if __name__ == '__main__':
    CORPUS_TRAIN = '../data/train.tsv'
    CORPUS_ETHALONE = '../data/ethalone.tsv'

    collector = EthaloneCorporaCollector()
    collector.collect_corpora(CORPUS_TRAIN, CORPUS_ETHALONE)
