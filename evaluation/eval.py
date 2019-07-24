from pathlib import Path

import pandas as pd

from rnn.diminutive_generator import DiminutiveGenerator


class DiminutiveEvaluator:
    def __init__(self, generator, path_to_corpus):
        assert isinstance(generator, DiminutiveGenerator)
        self.generator = generator
        self.ethalone_corpus = self.read_samples(path_to_corpus, ['name', 'dim_forms'])

    @staticmethod
    def read_samples(path_to_corpus, columns):
        with Path(path_to_corpus).open() as fin:
            names = (line.split() for line in fin.readlines())
        return pd.DataFrame(names, columns=columns)

    def evaluate(self, sample):
        # todo: calculate count of calls into self._find_default_transition() method! Maybe, decorator?
        correct, n = 0, 0
        for name in sample:
            dim = self.generator.generate_diminutive(name)
            if name in self.ethalone_corpus['names'] and dim in self.ethalone_corpus['dim_forms']:
                correct += 1
                n += 1
            elif name in self.ethalone_corpus['names']:
                n += 1
        return n, correct, correct / n
