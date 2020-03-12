from collections import Counter

import pymorphy2

import pandas as pd


class Annotator:
    def annotate(self, dataset):
        pass


class GenderAnnotator(Annotator):
    def __init__(self, output_path):
        self.morph = pymorphy2.MorphAnalyzer()
        self.output_path = output_path

    def __calculate_gender(self, name):
        tags = Counter(p.tag.gender for p in self.morph.parse(name)).most_common()
        if not tags or len(set(tags.values())) == 1:
            return 'neut'
        return tags[0][0]

    def annotate(self, dataset):
        dataset['Gender'] = dataset.Name.apply(lambda name: self.__calculate_gender(name))
        dataset.to_csv(self.output_path)
