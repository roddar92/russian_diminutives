import numpy as np
import sys

from scipy.stats import f, ttest_ind, sem, t, ks_2samp

from evaluation.eval import DiminutiveEvaluator
from rnn.diminutive_generator import DiminutiveGenerator
from utils.dim_io import read_samples


def get_acc_headers():
    return "{:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        '', 'Mean', 'Median', 'Std', 'Min', 'Max', 'CI 95% lower', 'CI 95% upper')


def get_headers():
    return "{:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        '', 'Mean', 'Median', 'Std', 'Min', 'Max', 'CI 95% lower', 'CI 95% upper')


def print_scores(label, headers, train_scores, test_scores, file=sys.stdout):
    print(label, file=file)
    print(headers, file=file)
    print('{:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}'
          .format('Train data', *compute_statistics(train_scores)), file=file)
    print('{:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}'
          .format('Test data', *compute_statistics(test_scores)), file=file)
    print(file=file)


def mean_confidence_interval(sample, confidence=0.95):
    n = len(sample)
    m, se = np.mean(sample), sem(sample)
    h = se * t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def f_test(a, b):
    a, b = np.array(a), np.array(b)
    score = np.var(a) / np.var(b)
    df1, df2 = len(a) - 1, len(b) - 1
    return score, f.cdf(score, df1, df2)


def compute_statistics(scores):
    scores = np.array(scores)
    m, lower, upper = mean_confidence_interval(scores)
    return (
        np.round(m, 5),
        np.round(np.median(scores), 5),
        np.round(np.std(scores), 5),
        np.round(np.min(scores), 5),
        np.round(np.max(scores), 5),
        np.round(lower, 5), np.round(upper, 5)
    )
    

def evaluate_stats_data(ethalone_path, train_path, train_sample, test_sample, ngram=2, times=100, fout=sys.stdout):
    gen = DiminutiveGenerator(ngram=ngram)
    gen.fit(train_path)

    evaluator = DiminutiveEvaluator(gen, ethalone_path)

    train_scores, train_euristics, test_scores, test_euristics = [], [], [], []
    train_pres, train_recalls, train_fscores = [], [], []
    test_pres, test_recalls, test_fscores = [], [], []
    for i in range(times):
        _, _, _, accuracy, euristics = evaluator.evaluate(np.unique(train_sample.name))
        train_scores.append(accuracy)
        train_euristics.append(euristics)
        _, _, _, accuracy, euristics = evaluator.evaluate(np.unique(test_sample.name))
        test_scores.append(accuracy)
        test_euristics.append(euristics)
        p, r, fs = evaluator.evaluate_precision_recall_fscore(np.unique(train_sample.name))
        train_pres.append(p), train_recalls.append(r), train_fscores.append(fs)
        p, r, fs = evaluator.evaluate_precision_recall_fscore(np.unique(test_sample.name))
        test_pres.append(p), test_recalls.append(r), test_fscores.append(fs)

        if i % 10 == 0:
            print(f'Processed {i} times...')

    print_scores(f'Evaluate generator with ngram={ngram}, {times} iterations',
                 get_acc_headers(), train_scores, test_scores, file=fout)

    print(file=fout)
    print(f'Euristics for train data: {np.round(np.mean(train_euristics), 5)}', file=fout)
    print(f'Euristics for test data: {np.round(np.mean(test_euristics), 5)}', file=fout)
    print(file=fout)

    print_scores('Precision', get_headers(), train_pres, test_pres, file=fout)
    print_scores('Recall', get_headers(), train_recalls, test_recalls, file=fout)
    print_scores('F-score', get_headers(), train_fscores, test_fscores, file=fout)

    return train_scores, test_scores


def check_hypothesis_for_ngrams(acc_bigram, accur_trigram, alpha=0.05, is_train=True, foutput=sys.stdout):
    label = 'train' if is_train else 'test'
    print(f'Test Hypothesis of variances and means equation for {label} data', file=foutput)
    score, pvalue = ks_2samp(acc_bigram, accur_trigram)
    print(f'Kolmogorov-Smirnov test for the same samples (score, p-value): {score}, {pvalue}', file=foutput)
    score, pvalue = f_test(acc_bigram, accur_trigram)
    print(f'F-test for variances (score, p-value): {score}, {pvalue}', file=foutput)
    equal_var = pvalue > alpha
    score, pvalue = ttest_ind(acc_bigram, accur_trigram, equal_var=equal_var)
    print(f'T-test score and p-value: {score}, {pvalue}', file=foutput)


if __name__ == '__main__':
    CORPUS_TRAIN = '../data/train.tsv'
    CORPUS_TEST = '../data/test.tsv'
    CORPUS_ETHALONE = '../data/ethalone.tsv'

    train = read_samples(CORPUS_TRAIN, ['name', 'dim'])
    test = read_samples(CORPUS_TEST, ['name'])

    samples = {}
    with open('stats_100.out', 'w', encoding='utf-8') as fo:
        train_label, test_label = 'train', 'test'
        for ngram_size in (2, 3):
            train_vals, test_vals = evaluate_stats_data(
                CORPUS_ETHALONE, CORPUS_TRAIN, train, test, ngram=ngram_size, fout=fo, times=100)
            samples[(ngram_size, train_label)] = train_vals
            samples[(ngram_size, test_label)] = test_vals
            print(file=fo)

        train_bigram, train_trigram = samples[(2, train_label)], samples[(3, train_label)]
        check_hypothesis_for_ngrams(train_bigram, train_trigram, alpha=0.05, foutput=fo)
        print(file=fo)

        test_bigram, test_trigram = samples[(2, test_label)], samples[(3, test_label)]
        check_hypothesis_for_ngrams(test_bigram, test_trigram, alpha=0.05, is_train=False, foutput=fo)
        print(file=fo)
