import pandas as pd

from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score

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
        # euristics - count of calls into self._find_default_transition() method with decorator
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
                # else:
                #     print(name, dim, base, dim_endings)
            total += 1
            euristics += int(flag)

        return total, correct, same, correct / total, euristics / total
    
    def is_diminutive_correct(self, name, diminutive):
        if diminutive == name:
            return False

        dims = self.ethalone_corpus['dim_form'][self.ethalone_corpus['name'] == name]
        if len(dims.values) > 0 and diminutive in eval(dims.values[0]):
            return True
        else:
            base, dim_endings = EthaloneCorporaCollector.get_possible_dim_engings(name)
            return any(diminutive == base + ending for ending in dim_endings)
    
    def evaluate_vocabulary_volume(self, sample, times=10):
        diminutive_vocabulary = defaultdict(set)
        
        for i in range(times):
            for name in sample:
                diminutive_vocabulary[name].add(self.generator.generate_diminutive(name))
            if i % 10 == 0:
                print(f'Processed {i} times...')
                
        print(f'Vocabulary volume with all forms is: {sum(len(l) for l in diminutive_vocabulary.values())}')
        print(f'Vocabulary volume with correct forms is: '
              f'{sum(self.is_diminutive_correct(n, w) for n, l in diminutive_vocabulary.items() for w in l)}')
        # return diminutive_vocabulary

    def evaluate_precision_recall_fscore(self, sample):
        y_true, y_pred = [1] * len(sample), []
        for name in sample:
            dim = self.generator.generate_diminutive(name)

            if dim == name:
                y_pred.append(0)
                continue

            dims = self.ethalone_corpus['dim_form'][self.ethalone_corpus['name'] == name]
            if len(dims.values) > 0 and dim in eval(dims.values[0]):
                y_pred.append(1)
            else:
                base, dim_endings = EthaloneCorporaCollector.get_possible_dim_engings(name)
                if any(dim == base + ending for ending in dim_endings):
                    y_pred.append(1)
                else:
                    y_pred.append(0)

        return precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)


def evaluate_data(ethalone_path, train_path, train_sample, test_sample, ngram=2):
    gen = DiminutiveGenerator(ngram=ngram)
    gen.fit(train_path)
    print(f'Evaluate generator with ngram={ngram}:')
    evaluator = DiminutiveEvaluator(gen, ethalone_path)

    print(f'Train data (total, correct, same forms, accuracy, % with used manual euristics): '
          f'{evaluator.evaluate(train_sample.name)}')
    print(f'Test data (total, correct, same forms, accuracy, % with used manual euristics): '
          f'{evaluator.evaluate(test_sample.name)}')

    score_title = 'Precision: {}, Recall: {}, F-score: {}'
    print('Train data:')
    print(score_title.format(*evaluator.evaluate_precision_recall_fscore(train_sample.name)))
    evaluator.evaluate_vocabulary_volume(train_sample.name)
    print('Test data:')
    print(score_title.format(*evaluator.evaluate_precision_recall_fscore(test_sample.name)))
    evaluator.evaluate_vocabulary_volume(test_sample.name)

    
if __name__ == '__main__':
    CORPUS_TRAIN = '../data/train.tsv'
    CORPUS_TEST = '../data/test.tsv'
    CORPUS_ETHALONE = '../data/ethalone.tsv'

    train_data = read_samples(CORPUS_TRAIN, ['name', 'dim'])
    test_data = read_samples(CORPUS_TEST, ['name'])

    for ngram_size in (2, 3):
        evaluate_data(CORPUS_ETHALONE, CORPUS_TRAIN, train_data, test_data, ngram=ngram_size)
        print()
