from evaluation.collect_ethalone import EthaloneCorporaCollector
from rnn.diminutive_generator import DiminutiveGenerator
from utils.dim_io import read_samples


class DiminutiveEvaluator:
    def __init__(self, generator, path_to_corpus):
        assert isinstance(generator, DiminutiveGenerator)
        self.generator = generator
        self.ethalone_corpus = read_samples(path_to_corpus, ['name', 'dim_forms'])

    def evaluate(self, sample):
        # todo: calculate count of calls into self._find_default_transition() method! Maybe, decorator?
        correct, total = 0, 0
        for name in sample:
            dim = self.generator.generate_diminutive(name)
            if name in self.ethalone_corpus['names'] and dim in self.ethalone_corpus['dim_forms']:
                correct += 1
                total += 1
            elif name in self.ethalone_corpus['names']:
                total += 1
            else:
                base, dim_endings = EthaloneCorporaCollector.get_possible_dim_engings(name)
                for ending in dim_endings:
                    if dim == base + ending:
                        correct += 1
                        total += 1
                        break

        return total, correct, correct / total
