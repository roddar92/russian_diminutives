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
        # todo: calculate count of calls into self._find_default_transition() method with decorator
        correct, same, euristics, total = 0, 0, 0, 0
        for name in sample:
            dim_flag = self.generator.generate_diminutive(name, print_euristic_flag=True)
            if type(dim_flag) == str:
                dim = dim_flag
            else:
                dim, flag = dim_flag
            
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
            euristics += int(flag)

        return total, correct, same, correct / total, euristics / total


def evaluate_data(ethalone_path, train_path, train_sample, test_sample, ngram=2):
    gen = DiminutiveGenerator(ngram=ngram)
    gen.fit(train_path)
    print(f'Evaluate generator with ngram={ngram}:')
    evaluator = DiminutiveEvaluator(gen, ethalone_path)

    print(f'Train data (total, correct, same forms, accuracy, % with used manual euristics): {evaluator.evaluate(train_sample.name)}')
    print(f'Test data (total, correct, same forms, accuracy, % with used manual euristics): {evaluator.evaluate(test_sample.name)}')

    
if __name__ == '__main__':
    CORPUS_TRAIN = '../data/train.tsv'
    CORPUS_TEST = '../data/test.tsv'
    CORPUS_ETHALONE = '../data/ethalone.tsv'

    train_sample = read_samples(CORPUS_TRAIN, ['name', 'dim'])
    test_sample = read_samples(CORPUS_TEST, ['name'])

    for ngram_size in (2, 3):
        evaluate_data(CORPUS_ETHALONE, CORPUS_TRAIN, train_sample, test_sample, ngram=ngram_size)
        print()
