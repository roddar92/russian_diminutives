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
            endings = ['уля', 'улечка', 'улик', 'уленька', 'утка', 'уточка', 'уня', 'унечка', 'унчик']
        elif name.endswith('ера'):
            base = name[:-1]
            endings = ['уля', 'улечка', 'улик', 'уленька', 'уня', 'унечка', 'унчик']
        elif name.lower().endswith('юля') or name.lower().endswith('оля') \
                or name.endswith('уся') or name.lower().endswith('ася'):
            base, endings = name[:-1], ['ечка', 'ик', 'енька', 'юшка', 'ютка', 'юта']
        elif name.lower().endswith('ас'):
            base, endings = name[:-1], ['ечка', 'ик', 'енька', 'юшка']
        elif name.lower().endswith('юта'):
            base, endings = name[:-1], ['очка', 'ик', 'ушка']
        elif name.lower().endswith('оня'):
            base, endings = name[:-1], ['ечка', 'юшка']
        elif name.endswith('иша'):
            base = name[:-1]
            endings = ['ечка', 'енька', 'уля', 'улечка', 'улик', 'уленька', 'утка', 'уточка']
        elif name.endswith('иса'):
            base = name[:-1]
            endings = ['очка', 'онька', 'ушка', 'уля', 'улечка', 'улик', 'уленька']
        elif name.lower().endswith('аня') or name.lower().endswith('адя') or name.endswith('атя'):
            base, endings = name[:-1], ['юша', 'юта', 'юха', 'ечка', 'енька']
        elif name.lower().endswith('рина'):
            base = name[:-2]
            endings = ['ша', 'шка', 'шечка', 'шенька']
        elif name.endswith('ава') or name.endswith('ара'):
            base = name[:-1]
            endings = ['очка', 'ик', 'ушка']
        elif name.endswith('ата') or name.endswith('ита'):
            base = name[:-1]
            endings = ['очка', 'ик', 'ушка', 'уля', 'улечка', 'улик', 'уленька', 'уся']
        elif name.endswith('ина') or name.lower().endswith('ида'):
            base = name[:-1]
            endings = ['очка', 'ушка', 'уша', 'уля', 'уся']
        elif name.endswith('оша'):
            base = name[:-1]
            endings = ['ечка', 'ик', 'енька']
        elif name.endswith('еня'):
            base = name[:-1]
            endings = ['ечка', 'ютка', 'юта', 'юленька', 'юлечка', 'юля', 'юшка', 'юша']
        elif name.endswith('аля'):
            base = name[:-1]
            endings = ['ечка', 'ик', 'енька', 'юля', 'юша']
        elif name.endswith('ова'):
            base = name[:-1]
            endings = ['очка', 'ушка']
        elif name.endswith('ана') or name.endswith('ада'):
            base = name[:-1]
            endings = ['очка', 'ушка', 'уля', 'уся']
        elif name.endswith('ль') or name.endswith('уша') or name.endswith('юша'):
            base = name[:-1]
            endings = ['ечка', 'енька']
        elif name.endswith('ан'):
            base = name
            endings = ['чик', 'уля', 'ушка']
        elif name.endswith('ха'):
            base = name[:-2]
            endings = ['ша', 'шка', 'шечка', 'шенька']
        elif name.endswith('ая') or name.endswith('ея') or name.endswith('еля') \
                or name.endswith('ей') or name.endswith('ай'):
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
