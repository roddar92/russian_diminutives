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
        real_name, diminutive = real_name.lower(), f'{diminutive.lower()}'
        n_chars = start * ngram
        max_len = max(len(real_name), len(diminutive))
        for i in range(max_len):
            if i < len(real_name):
                ch, dim_ch = real_name[i], diminutive[i]
                if ch != dim_ch:
                    diminutive_transits[f'{n_chars}{ch}'].add(diminutive[i:])
                    break
                else:
                    next_char = real_name[i]
                    n_chars = n_chars[1:] + next_char
            else:
                if i == len(real_name) and real_name.endswith(n_chars):
                    ch = '$'
                    diminutive_transits[f'{n_chars}{ch}'].add(diminutive[i:])
                    break
                else:
                    break

    return diminutive_transits


def simplify_suffixes(dist):
    union = [dim_set for ngram, dim_set in dist.items() if ngram in ('аша', 'еша', 'иша')]
    dist['ша'] = set().union(*union)
    del dist['аша']
    del dist['еша']
    del dist['иша']
    
    union = [dim_set for ngram, dim_set in dist.items() if ngram in ('аня', 'еня', 'иня', 'уня')]
    dist['ня'] = set().union(*union)
    del dist['аня']
    del dist['еня']
    del dist['иня']
    del dist['уня']
    
    union = [dim_set for ngram, dim_set in dist.items() if ngram in ('ася', 'еся', 'ися', 'уся')]
    dist['ся'] = set().union(*union)
    del dist['ася']
    del dist['еся']
    del dist['ися']
    del dist['уся']
    
    union = [dim_set for ngram, dim_set in dist.items() if ngram in ('атя', 'етя', 'итя')]
    dist['тя'] = set().union(*union)
    del dist['атя']
    del dist['етя']
    del dist['итя']
    
    union = [dim_set for ngram, dim_set in dist.items() if ngram in ('ана', 'ена', 'ина')]
    dist['на'] = set().union(*union)
    del dist['ана']
    del dist['ена']
    del dist['ина']
    
    union = [dim_set for ngram, dim_set in dist.items() if ngram in ('аса', 'еса', 'иса')]
    dist['са'] = set().union(*union)
    del dist['аса']
    del dist['еса']
    del dist['иса']
    
    union = [dim_set for ngram, dim_set in dist.items() if ngram in ('ата', 'ета', 'ита')]
    dist['та'] = set().union(*union)
    del dist['ата']
    del dist['ета']
    del dist['ита']
    
    union = [dim_set for ngram, dim_set in dist.items() if ngram in ('ай$', 'ей$', 'ий$')]
    dist['й$'] = set().union(*union)
    del dist['ай$']
    del dist['ей$']
    del dist['ий$']
    
    
    union = [dim_set for ngram, dim_set in dist.items() if ngram.endswith('я') and ngram[-2] in 'аеи']
    dist['<vowel>й'] = set().union(*union)
    
    for ngram in dist.keys():
        if ngram.endswith('я') and ngram[-2] in 'аеи':
            del dist[ngram]
            
            
    union = [dim_set for ngram, dim_set in dist.items() if ngram.endswith('$') and ngram[-2] not in 'аеиоуя']
    dist['<non-vowel>$'] = set().union(*union)
    
    for ngram in dist.keys():
        if ngram.endswith('$') and ngram[-2] not in 'аеиоуя':
            del dist[ngram]
    
    return dist


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
    plot_suffixes_dustribution(simplify_suffixes(dist))

