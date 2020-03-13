import pymorphy2

from collections import Counter

from utils.dim_io import read_samples


class Annotator:
    def annotate(self, dataset):
        pass


class GenderAnnotator(Annotator):
    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()

    def __calculate_gender(self, name):
        tags = Counter(p.tag.gender for p in self.morph.parse(name)).most_common()
        if not tags or (len(tags) > 1 and len(set(dict(tags).values())) == 1):
            return 'neut'
        return tags[0][0] or 'femn'

    def annotate(self, dataset):
        dataset['Gender'] = dataset.Name.apply(lambda name: self.__calculate_gender(name))
        return dataset

    @staticmethod
    def save_annotation(dataset, output_path):
        dataset.to_csv(output_path, sep='\t', index=False)


if __name__ == '__main__':
    CORPUS_TRAIN = '../data/train.tsv'
    CORPUS_TRAIN_OUT = '../data/train-gender.tsv'
    CORPUS_TEST = '../data/test.tsv'
    CORPUS_TEST_OUT = '../data/test-gender.tsv'

    DATASET_CONFIG = {
        CORPUS_TRAIN: {
            'columns': ['Name', 'Diminutive'],
            'output_path': CORPUS_TRAIN_OUT
        },
        CORPUS_TEST: {
            'columns': ['Name'],
            'output_path': CORPUS_TEST_OUT
        },
    }

    annotator = GenderAnnotator()
    for input_path, configs in DATASET_CONFIG.items():
        df = read_samples(input_path, columns=configs['columns'])
        df = annotator.annotate(df)
        annotator.save_annotation(df, configs['output_path'])
