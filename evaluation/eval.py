import pandas as pd

from evaluation.collect_ethalone import EthaloneCorporaCollector
from rnn.diminutive_generator import DiminutiveGenerator
from utils.dim_io import read_samples


class DiminutiveEvaluator:
    def __init__(self, dim_generator, path_to_corpus):
        assert isinstance(dim_generator, DiminutiveGenerator)
        self.generator = dim_generator
        self.ethalone_corpus = pd.read_csv(path_to_corpus, sep=',', usecols=['0', '1'])
        self.ethalone_corpus.columns = ['name', 'dim_form']

    def evaluate(self, sample):
        # todo: calculate count of calls into self._find_default_transition() method! Maybe, decorator?
        correct, same, total = 0, 0, 0
        for name in sample:
            dim = self.generator.generate_diminutive(name)
            
            if dim == name:
                same += 1
                total += 1
                continue
            
            dims = self.ethalone_corpus['dim_form'][self.ethalone_corpus['name'] == name]
            if len(dims.values) > 0 and dim in eval(dims.values[0]):
                correct += 1
            else:
                base, dim_endings = EthaloneCorporaCollector.get_possible_dim_engings(name)
                if any(dim == base + ending for ending in dim_endings):
                    correct += 1
                #else:
                #    print(name, dim, base, dim_endings)
            total += 1

        return total, correct, correct / total, same


if __name__ == '__main__':
    CORPUS_TRAIN = '../data/train.tsv'
    CORPUS_TEST = '../data/test.tsv'
    CORPUS_ETHALONE = '../data/ethalone.tsv'

    train_sample = read_samples(CORPUS_TRAIN, ['name', 'dim'])
    test_sample = read_samples(CORPUS_TEST, ['name'])

    gen1 = DiminutiveGenerator(ngram=2)
    gen1.fit(CORPUS_TRAIN)

    print('Evaluate generator with bigrams:')
    evaluator = DiminutiveEvaluator(gen1, CORPUS_ETHALONE)

    print('Train data (total, correct, accuracy, same):', evaluator.evaluate(train_sample.name))
    print('Test data (total, correct, accuracy, same):', evaluator.evaluate(test_sample.name))
    print()

    gen2 = DiminutiveGenerator(ngram=3)
    gen2.fit(CORPUS_TRAIN)

    print('Evaluate generator with trigrams:')
    evaluator = DiminutiveEvaluator(gen2, CORPUS_ETHALONE)

    print('Train data (total, correct, accuracy, same):', evaluator.evaluate(train_sample.name))
    print('Test data (total, correct, accuracy, same):', evaluator.evaluate(test_sample.name))
