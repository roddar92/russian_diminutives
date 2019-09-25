import numpy as np
import sys

from scipy.stats import sem, t

from evaluation.eval import DiminutiveEvaluator
from rnn.diminutive_generator import DiminutiveGenerator
from utils.dim_io import read_samples


def get_headers():
    return "{:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format('', 'Mean', 'Median', 'Std', 'Min', 'Max', 'CI 95%')


def mean_confidence_interval(sample, confidence=0.95):
    n = len(sample)
    m, se = np.mean(sample), sem(sample)
    h = se * t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def compute_statistics(scores):
    scores = np.array(scores)
    m, lower, upper = mean_confidence_interval(scores)
    return (
        np.round(m, 5),
        np.round(np.median(scores), 5),
        np.round(np.std(scores), 5),
        np.round(np.min(scores), 5),
        np.round(np.max(scores), 5),
        (np.round(lower, 5), np.round(upper, 5))
    )
    

def evaluate_stats_data(ethalone_path, train_path, train_sample, test_sample, ngram=2, times=100, fout=sys.stdout):
    gen = DiminutiveGenerator(ngram=ngram)
    gen.fit(train_path)

    evaluator = DiminutiveEvaluator(gen, ethalone_path)
    print(f'Evaluate generator with ngram={ngram}, {times} iterations', file=fout)
    print(get_headers(), file=fout)

    train_scores, train_euristics, test_scores, test_euristics = [], [], [], []
    for i in range(times):
        _, _, _, accuracy, euristics = evaluator.evaluate(train_sample.name)
        train_scores.append(accuracy)
        train_euristics.append(euristics)
        _, _, _, accuracy, euristics = evaluator.evaluate(test_sample.name)
        test_scores.append(accuracy)
        test_euristics.append(euristics)

        if i % 10 == 0:
            print(f'Processed {i} times...')

    print('{:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}'
          .format('Train data', *compute_statistics(train_scores)), file=fout)
    print('{:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}'
          .format('Test data', *compute_statistics(test_scores)), file=fout)

    print(file=fout)
    print(f'Euristics for train data: {np.round(np.mean(train_euristics), 5)}', file=fout)
    print(f'Euristics for test data: {np.round(np.mean(test_euristics), 5)}', file=fout)
    print(file=fout)


if __name__ == '__main__':
    CORPUS_TRAIN = '../data/train.tsv'
    CORPUS_TEST = '../data/test.tsv'
    CORPUS_ETHALONE = '../data/ethalone.tsv'

    train = read_samples(CORPUS_TRAIN, ['name', 'dim'])
    test = read_samples(CORPUS_TEST, ['name'])

    with open('stats_100.out', 'w', encoding='utf-8') as fout:
        for ngram_size in (2, 3):
            evaluate_stats_data(CORPUS_ETHALONE, CORPUS_TRAIN, train, test, ngram=ngram_size, fout=fout, times=100)
            print(file=fout)
