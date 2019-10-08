import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from collections import defaultdict

from utils.dim_io import read_samples


def get_diminutive_suffixes(dataset, ngram=2):
    start = '*'

    print('Collecting of diminutive suffixes...')
    diminutive_transits = defaultdict(set)

    for real_name, diminutive in zip(dataset.Name, dataset.Diminutive):
        real_name, diminutive = real_name.lower(), f'{diminutive.lower()}$'
        n_chars = start * ngram
        stay_within_name = True
        max_len = max(len(real_name), len(diminutive))
        for i in range(max_len):
            if i < len(real_name) and stay_within_name:
                ch, dim_ch = real_name[i], diminutive[i]
                if ch != dim_ch:
                    diminutive_transits[n_chars].add(diminutive[i:-1])
                    stay_within_name = False
                    break
                else:
                    next_char = real_name[i]
                    n_chars = n_chars[1:] + next_char
            else:
                """if i == len(real_name) and real_name.endswith(n_chars):
                    ch = '$'
                    diminutive_transits[n_chars].add(diminutive[i:-1])
                else:
                    break"""

    return diminutive_transits


def plot_suffixes_dustribution(dist):
    labels, counts = [], []
    for ngram, dimin_suffixes in dist.items():
        labels.append(ngram)
        counts.append(len(dimin_suffixes))

    df_viz = pd.DataFrame({'ngram': labels, 'suffix count': counts})
    sns.set_color_codes("pastel")
    sns.barplot(x='suffix count', y='ngram', data=df_viz, color='g')
    sns.despine(left=True, bottom=True)
    plt.show()



if __name__ == '__main__':
    PATH_TO_SAMPLES = '../data/train.tsv'

    df = read_samples(PATH_TO_SAMPLES, columns=['Name', 'Diminutive'])
    dist = get_diminutive_suffixes(df)
    plot_suffixes_dustribution(dist)

