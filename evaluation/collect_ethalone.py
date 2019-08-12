import pandas as pd

from collections import defaultdict

from rnn.diminutive_generator import DiminutiveGenerator
from utils.dim_io import read_samples


class EthaloneCorporaCollector:
    COL_NAMES = ['name', 'dim_forms']

    def __init__(self):
        self.ethalone = defaultdict(list)

    @staticmethod
    def get_possible_dim_engings(name):
        name = DiminutiveGenerator.normalize_k_suffix(name)

        if name.endswith('аша') or name.endswith('уля'):
            base = name[:-1] if name.endswith('аша') else name[:-3]
            endings = ['енька', 'ечка', 'уля', 'улечка', 'улик', 'уленька',
                       'утка', 'ута', 'уха', 'уточка', 'уня', 'унечка', 'унчик']
        elif name.endswith('ера'):
            base = name[:-1]
            endings = ['очка', 'онька', 'уля', 'улечка', 'улик', 'уленька', 'уня', 'ушка', 'унечка', 'унчик']
        elif name.lower().endswith('юля') or name.lower().endswith('оля') \
                or name.endswith('уся') or name.lower().endswith('ася') or name.lower().endswith('аря'):
            base, endings = name[:-1], ['ечка', 'ик', 'енька', 'юшка', 'юня', 'ютка', 'юта']
        elif name.endswith('итя'):
            base, endings = name[:-1], ['ечка', 'юха', 'енька', 'юшка', 'юша', 'юня', 'юшечка', 'юшенька']
        elif name.lower().endswith('ас'):
            base, endings = name, ['ечка', 'ик', 'енька', 'юшка']
        elif name.lower().endswith('юта') or name.lower().endswith('ева'):
            base, endings = name[:-1], ['очка', 'ик', 'ушка']
        elif name.lower().endswith('оня'):
            base, endings = name[:-1], ['ечка', 'юшка']
        elif name.endswith('иша'):
            base = name[:-1]
            endings = ['ечка', 'енька', 'уля', 'уня', 'улечка', 'улик', 'уленька', 'ута', 'утка', 'уточка']
        elif name.endswith('иса'):
            base = name[:-1]
            endings = ['очка', 'онька', 'ушка', 'уля', 'улечка', 'улик', 'уленька']
        elif name.lower().endswith('аня') or name.lower().endswith('адя') or \
                name.endswith('атя') or name.endswith('стя'):
            base, endings = name[:-1], ['юня', 'юша', 'юшка', 'юсик', 'ик',
                                        'юнечка', 'юшечка', 'юшенька', 'юта', 'юха', 'ечка', 'енька']
        elif name.lower().endswith('рина'):
            base = name[:-2]
            endings = ['ша', 'шка', 'шечка', 'шенька', 'ночка']
        elif name.endswith('ава') or name.endswith('ара') or name.endswith('нна'):
            base = name[:-1]
            endings = ['очка', 'ик', 'ушка', 'уня', 'уся', 'усик']
        elif name.endswith('ата') or name.endswith('ита'):
            base = name[:-1]
            endings = ['очка', 'ик', 'ушка', 'уля', 'улечка', 'улик', 'уленька', 'уся', 'усик']
        elif name.endswith('сса'):
            base = name[:-1]
            endings = ['очка', 'ик', 'ушка', 'онька']
        elif name.endswith('ина') or name.lower().endswith('ида') or name.lower().endswith('ила') or \
                name.lower().endswith('она'):
            base = name[:-1]
            endings = ['очка', 'ушка', 'уша', 'уля', 'уся']
        elif name.endswith('оша') or name.endswith('ося'):
            base = name[:-1]
            endings = ['ечка', 'ик', 'енька']
        elif name.endswith('еня'):
            base = name[:-1]
            endings = ['ечка', 'ютка', 'юта', 'юленька', 'юлечка', 'юшечка', 'юля', 'юшка', 'юша']
        elif name.lower().endswith('аля'):
            base = name[:-1]
            endings = ['ечка', 'ик', 'енька', 'юня', 'юнечка', 'юша', 'ёк']
        elif name.lower().endswith('отя'):
            base = name[:-1]
            endings = ['ечка', 'ик', 'енька']
        elif name.endswith('ова') or name.lower().endswith('ива'):
            base = name[:-1]
            endings = ['очка', 'ушка']
        elif name.endswith('ура'):
            base = name[:-1]
            endings = ['очка', 'ик']
        elif name.endswith('ана') or name.endswith('ада') or \
                name.endswith('ена') or name.endswith('ета') or name.endswith('ора'):
            base = name[:-1]
            endings = ['очка', 'ушка', 'уля', 'уся', 'усик']
        elif name.endswith('уня'):
            base = name[:-1]
            endings = ['енька', 'ечка', 'яша', 'яшка']
        elif name.endswith('ль') or name.endswith('уша') or \
                name.lower().endswith('юша') or name.lower().endswith('юся'):
            base = name[:-1]
            endings = ['ечка', 'енька', 'ик']
        elif name.endswith('ика'):
            base = name[:-1]
            endings = ['уся', 'усик', 'уля', 'уня', 'ушка']
        elif name.endswith('ан') or name.endswith('он') or name.endswith('ян') or \
                name.endswith('ип') or name.endswith('ап')or name.endswith('ат'):
            base = name
            endings = ['чик', 'уля', 'ушка', 'ик']
        elif name.endswith('ст') or name.endswith('рт'):
            base = name
            endings = ['ушка', 'ик']
        elif name.endswith('има') or name.endswith('ира') or name.lower().endswith('яна'):
            base = name[:-1]
            endings = ['ушка', 'очка', 'уля']
        elif name.endswith('ста') or name.endswith('рта'):
            base = name[:-1]
            endings = ['ушка', 'очка']
        elif name.endswith('иза'):
            base = name[:-1]
            endings = ['ушка', 'очка', 'онька']
        elif name.endswith('еша'):
            base = name[:-1]
            endings = ['ечка', 'енька', 'унчик', 'уня', 'уля', 'улечка', 'уленька']
        elif name.endswith('ёна') or name.endswith('ёпа'):
            base = name[:-1]
            endings = ['чик', 'очка', 'ушка']
        elif name.endswith('им') or name.endswith('ор') or name.endswith('ен'):
            base = name
            endings = ['чик', 'уша', 'очка', 'ушка', 'уля']
        elif name.endswith('ха'):
            base = name[:-2]
            endings = ['ша', 'шка', 'шечка', 'хочка', 'шенька', 'на', 'та', 'тка', 'сик', 'ся']
        elif name.endswith('ая') or name.endswith('ея') or name.endswith('еля') \
                or name.endswith('ей') or name.endswith('ай'):
            base = name[:-1]
            endings = ['ечка', 'енька', 'юшка', 'юша', 'юшечка', 'юшенька', 'йка']
        elif name.endswith('рь'):
            base = name
            endings = ['ёк', 'ёчек']
        elif name.endswith('рж'):
            base = name
            endings = ['ик']
        else:
            base, endings = name[:-1] if name[-1] in 'аеёиоуыэюяь' else name, []
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
