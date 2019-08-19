import numpy as np
import pandas as pd

from evaluation.eval import DiminutiveEvaluator
from rnn.diminutive_generator import DiminutiveGenerator
from utils.dim_io import read_samples


def compute_statistics(scores):
    scores = np.array(scores)
    return {
        "Mean: " : np.mean(train_scores, axis=0),
        "Median: ": np.median(train_scores, axis=0),
        "Std: ": np.std(train_scores, axis=0),
        "Min: ": np.min(train_scores, axis=0),
        "Max: ": np.max(train_scores, axis=0)
    }
    

def evaluate_stats_data(ethalone_path, train_path, train_sample, test_sample, ngram=2, times=100):
    gen = DiminutiveGenerator(ngram=ngram)
    gen.fit(train_path)
    print(f'Evaluate generator with ngram={ngram}:')
    evaluator = DiminutiveEvaluator(gen, ethalone_path)
    
    train_scores, test_scores = [], []
    
    for _ in range(times):
        _, _, _, accuracy, euristics = evaluator.evaluate(train_sample.name)
        train_scores.append([accuracy, euristics])
        _, _, _, accuracy, euristics = evaluator.evaluate(test_sample.name)
        test_scores.append([accuracy, euristics])

    print(f'Train data (Statistics for accuracy & euristics, %): {compute_statistics(train_scores)}')
    print(f'Test data (Statistics for accuracy & euristics, %): {compute_statistics(test_scores)}')


if __name__ == '__main__':
    CORPUS_TRAIN = '../data/train.tsv'
    CORPUS_TEST = '../data/test.tsv'
    CORPUS_ETHALONE = '../data/ethalone.tsv'

    train_sample = read_samples(CORPUS_TRAIN, ['name', 'dim'])
    test_sample = read_samples(CORPUS_TEST, ['name'])

    for ngram_size in (2, 3):
        evaluate_stats_data(CORPUS_ETHALONE, CORPUS_TRAIN, train_sample, test_sample, ngram=ngram_size)
        print()
