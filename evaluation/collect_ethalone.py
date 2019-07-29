import pandas as pd

from collections import defaultdict

from utils.dim_io import read_samples


class EthaloneCorporaCollector:
    COL_NAMES = ['name', 'dim_forms']

    def __init__(self):
        self.ethalone = defaultdict(list)

    @staticmethod
    def get_possible_dim_engings(name):
        if name.endswith('аша') or name.endswith('уля'):
            base = name[:-1] if name.endswith('аша') else name[:-3]
            endings = ['уля', 'улечка', 'улик', 'уленька', 'утка', 'уточка']
        elif name.lower().endswith('юля') or name.lower().endswith('оля') \
                or name.endswith('уся') or name.endswith('ася'):
            base, endings = name[:-1], ['ечка', 'ик', 'енька', 'юшка']
        elif name.endswith('иша'):
            base = name[:-1]
            endings = ['ечка', 'енька', 'уля', 'улечка', 'улик', 'уленька', 'утка', 'уточка']
        elif name.lower().endswith('аня') or name.lower().endswith('адя') or name.endswith('атя'):
            base, endings = name[:-1], ['юша', 'юта', 'юха', 'ечка', 'енька']
        elif name.lower().endswith('рина'):
            base = name[:-2]
            endings = ['ша', 'шка', 'шечка', 'шенька']
        elif name.endswith('ава') or name.endswith('ара'):
            base = name[:-1]
            endings = ['очка', 'ик', 'ушка']
        elif name.endswith('ата'):
            base = name[:-1]
            endings = ['очка', 'ик', 'ушка', 'уля', 'улечка', 'улик', 'уленька', 'уся']
        elif name.endswith('ина'):
            base = name[:-1]
            endings = ['очка', 'ушка', 'уля', 'уся']
        elif name.endswith('оша'):
            base = name[:-1]
            endings = ['ечка', 'ик', 'енька']
        elif name.endswith('ова'):
            base = name[:-1]
            endings = ['очка', 'ушка']
        elif name.endswith('ана'):
            base = name[:-1]
            endings = ['очка', 'ушка', 'уля']
        elif name.endswith('ая'):
            base = name[:-1]
            endings = ['ечка', 'енька', 'юшка', 'юшечка', 'юшенька']
        else:
            base, endings = name[:-1], []
        return base, endings

    def collect_corpora(self, path_to_corpus, path_to_output_corpus):
        for _, row in read_samples(path_to_corpus, ['name', 'diminutive']).iterrows():
            self.ethalone[row['name']].append(row['diminutive'])
            base, endings = self.get_possible_dim_engings(row['name'])

            for ending in endings:
                if base + ending not in self.ethalone[row['name']]:
                    self.ethalone[row['name']].append(base + ending)

        pd.DataFrame(list(self.ethalone.items()), columns=self.COL_NAMES).to_csv(path_to_output_corpus)



if __name__ == '__main__':
    CORPUS_TRAIN = '../data/train.tsv'
    CORPUS_ETHALONE = '../data/ethalone.tsv'

    collector = EthaloneCorporaCollector()
    collector.collect_corpora(CORPUS_TRAIN, CORPUS_ETHALONE)
