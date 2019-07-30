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
        correct, total = 0, 0
        for name in sample:
            dim = self.generator.generate_diminutive(name)
            if name in self.ethalone_corpus['name'] and dim in self.ethalone_corpus['dim_form']:
                correct += 1
                total += 1
            elif name in self.ethalone_corpus['name']:
                total += 1
            else:
                base, dim_endings = EthaloneCorporaCollector.get_possible_dim_engings(name)
                for ending in dim_endings:
                    if dim == base + ending:
                        correct += 1
                        break
                total += 1

        return total, correct, correct / total


if __name__ == '__main__':
    CORPUS_TRAIN = '../data/train.tsv'
    CORPUS_TEST = '../data/test.tsv'
    CORPUS_ETHALONE = '../data/ethalone.tsv'

    generator = DiminutiveGenerator(ngram=2)
    generator.fit(CORPUS_TRAIN)

    print('Evaluate generator with bigrams:')
    evaluator = DiminutiveEvaluator(generator, CORPUS_ETHALONE)
    train_sample = read_samples(CORPUS_TRAIN, ['name', 'dim'])
    print('Train data (total, correct, accuracy):', evaluator.evaluate(train_sample.name))
    test_sample = read_samples(CORPUS_TEST, ['name'])
    print('Test data (total, correct, accuracy):', evaluator.evaluate(test_sample.name))
    print()

    generator = DiminutiveGenerator(ngram=3)
    generator.fit(CORPUS_TRAIN)

    print('Evaluate generator with trigrams:')
    evaluator = DiminutiveEvaluator(generator, CORPUS_ETHALONE)
    train_sample = read_samples(CORPUS_TRAIN, ['name', 'dim'])
    print('Train data (total, correct, accuracy):', evaluator.evaluate(train_sample.name))
    test_sample = read_samples(CORPUS_TEST, ['name'])
    print('Test data (total, correct, accuracy):', evaluator.evaluate(test_sample.name))
