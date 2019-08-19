import numpy as np
import sys

from evaluation.eval import DiminutiveEvaluator
from rnn.diminutive_generator import DiminutiveGenerator
from utils.dim_io import read_samples


def compute_statistics(scores):
    scores = np.array(scores)
    return {
        "Mean: ": np.mean(scores, axis=0),
        "Median: ": np.median(scores, axis=0),
        "Std: ": np.std(scores, axis=0),
        "Min: ": np.min(scores, axis=0),
        "Max: ": np.max(scores, axis=0)
    }
    

def evaluate_stats_data(ethalone_path, train_path, train_sample, test_sample, ngram=2, times=100, fout=sys.stdout):
    gen = DiminutiveGenerator(ngram=ngram)
    gen.fit(train_path)

    evaluator = DiminutiveEvaluator(gen, ethalone_path)
    print(f'Evaluate generator with ngram={ngram}:', file=fout)

    train_scores, test_scores = [], []
    for i in range(times):
        _, _, _, accuracy, euristics = evaluator.evaluate(train_sample.name)
        train_scores.append([accuracy, euristics])
        _, _, _, accuracy, euristics = evaluator.evaluate(test_sample.name)
        test_scores.append([accuracy, euristics])

        if i % 10 == 0:
            print(f'Processed {i} times...')

    print(f'Train data (Statistics for accuracy & euristics, %): {compute_statistics(train_scores)}', file=fout)
    print(f'Test data (Statistics for accuracy & euristics, %): {compute_statistics(test_scores)}', file=fout)


if __name__ == '__main__':
    CORPUS_TRAIN = '../data/train.tsv'
    CORPUS_TEST = '../data/test.tsv'
    CORPUS_ETHALONE = '../data/ethalone.tsv'

    train = read_samples(CORPUS_TRAIN, ['name', 'dim'])
    test = read_samples(CORPUS_TEST, ['name'])

    with open('stats.out', 'w', encoding='utf-8') as fout:
        for ngram_size in (2, 3):
            evaluate_stats_data(CORPUS_ETHALONE, CORPUS_TRAIN, train, test, ngram=ngram_size, fout=fout)
            print(file=fout)
