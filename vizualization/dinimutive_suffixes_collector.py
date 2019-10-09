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
                    diminutive_transits[f'{n_chars}{real_name[i:]}'].add(diminutive[i:])
                    break
                else:
                    next_char = real_name[i]
                    n_chars = n_chars[1:] + next_char
            else:
                if i == len(real_name) and real_name.endswith(n_chars):
                    diminutive_transits[f'{n_chars}$'].add(diminutive[i:])
                    break
                else:
                    break

    return diminutive_transits


def simplify_suffixes(dist):
    union = [dim_set for ngram, dim_set in dist.items() if ngram in ('аша', 'еша', 'иша')]
    dist['(а|е|и)ша'] = set().union(*union)
    del dist['аша']
    del dist['еша']
    del dist['иша']

    union = [dim_set for ngram, dim_set in dist.items() if ngram in ('оша', 'юша', 'яша')]
    dist['(о|ю|я)ша'] = set().union(*union)
    del dist['оша']
    del dist['юша']
    del dist['яша']

    union = [dim_set for ngram, dim_set in dist.items() if ngram in ('ава', 'ева', 'ива', 'ова')]
    dist['(а|е|и|о)ва'] = set().union(*union)
    del dist['ава']
    del dist['ева']
    del dist['ива']
    del dist['ова']

    union = [dim_set for ngram, dim_set in dist.items() if ngram in ('аля', 'еля', 'ёля', 'иля', 'оля', 'уля', 'юля')]
    dist['<vowel>ля'] = set().union(*union)
    del dist['аля']
    del dist['еля']
    del dist['ёля']
    del dist['иля']
    del dist['оля']
    del dist['уля']
    del dist['юля']

    union = [dim_set for ngram, dim_set in dist.items() if ngram in ('аня', 'еня', 'ёня', 'иня', 'уня')]
    dist['<vowel>ня'] = set().union(*union)
    del dist['аня']
    del dist['еня']
    del dist['ёня']
    del dist['иня']
    del dist['уня']
    
    union = [dim_set for ngram, dim_set in dist.items()
             if ngram in ('ася', 'еся', 'ёся', 'ися', 'ося', 'уся', 'юся', 'яся')]
    dist['<vowel>ся'] = set().union(*union)
    del dist['ася']
    del dist['еся']
    del dist['ёся']
    del dist['ися']
    del dist['ося']
    del dist['уся']
    del dist['юся']
    del dist['яся']

    union = [dim_set for ngram, dim_set in dist.items() if ngram in ('атя', 'етя', 'итя', 'отя', 'утя')]
    dist['<vowel>тя'] = set().union(*union)
    del dist['атя']
    del dist['етя']
    del dist['итя']
    del dist['отя']
    del dist['утя']

    union = [dim_set for ngram, dim_set in dist.items()
             if ngram in ('ана', 'ена', 'ёна', 'йна', 'ина', 'нна', 'уна', 'юна')]
    dist['<vowel>на'] = set().union(*union)
    del dist['ана']
    del dist['ена']
    del dist['йна']
    del dist['ёна']
    del dist['ина']
    del dist['нна']
    del dist['уна']
    del dist['юна']

    union = [dim_set for ngram, dim_set in dist.items() if ngram in ('ара', 'ера', 'ира', 'ора', 'ура')]
    dist['<vowel>ра'] = set().union(*union)
    del dist['ара']
    del dist['ера']
    del dist['ира']
    del dist['ора']
    del dist['ура']
    
    union = [dim_set for ngram, dim_set in dist.items() if ngram in ('аса', 'еса', 'иса')]
    dist['<vowel>са'] = set().union(*union)
    del dist['аса']
    del dist['еса']
    del dist['иса']
    
    union = [dim_set for ngram, dim_set in dist.items() if ngram in ('ата', 'ета', 'ита', 'юта')]
    dist['<vowel>та'] = set().union(*union)
    del dist['ата']
    del dist['ета']
    del dist['ита']
    del dist['юта']

    union = [dim_set for ngram, dim_set in dist.items() if ngram in ('ель', 'иль', 'оль')]
    dist['<vowel>ль'] = set().union(*union)
    del dist['ель']
    del dist['иль']
    del dist['оль']

    union = [dim_set for ngram, dim_set in dist.items() if ngram in ('арь', 'орь')]
    dist['<vowel>рь'] = set().union(*union)
    del dist['арь']
    del dist['орь']

    union = [dim_set for ngram, dim_set in dist.items() if ngram.endswith('й') and ngram[-2] in 'аеи']
    dist['<vowel>й'] = set().union(*union)

    for ngram in list(dist.keys()):
        if ngram.endswith('й') and ngram[-2] in 'аеи':
            del dist[ngram]

    union = [dim_set for ngram, dim_set in dist.items() if ngram.endswith('я') and ngram[-2] in 'аеиоу']
    dist['<vowel>я'] = set().union(*union)
    
    for ngram in list(dist.keys()):
        if ngram.endswith('я') and ngram[-2] in 'аеиоу':
            del dist[ngram]
            
            
    union = [dim_set for ngram, dim_set in dist.items() if ngram.endswith('$') and ngram[-2] not in 'аеиоуя']
    dist['<non-vowel>$'] = set().union(*union)
    
    for ngram in list(dist.keys()):
        if ngram.endswith('$') and ngram[-2] not in 'аеиоуя':
            del dist[ngram]
            
    union = [dim_set for ngram, dim_set in dist.items() if ngram.startswith('ь') and ngram[-2] not in 'аеиоуя']
    dist['ь<non-vowel>'] = set().union(*union)
    
    for ngram in list(dist.keys()):
        if ngram.startswith('ь') and ngram[-2] not in 'аеиоуя':
            del dist[ngram]

    union = [dim_set for ngram, dim_set in dist.items() if ngram.endswith('ха')]
    dist['<vowel>й'] = set().union(*union)

    for ngram in list(dist.keys()):
        if ngram.endswith('ха'):
            del dist[ngram]


def plot_top_suffixes(data, topn=20):
    labels, counts = [], []
    for ngram, dimin_suffixes in sorted(data.items(), key=lambda x: -len(x[-1])):
        labels.append(ngram)
        counts.append(len(dimin_suffixes))

    df_viz = pd.DataFrame({'last letters': labels[:topn], 'suffix count': counts[:topn]})
    sns.set_color_codes("pastel")
    sns.barplot(x='suffix count', y='last letters', data=df_viz, color='g')
    sns.despine(left=True, bottom=True)
    plt.show()


def plot_suffixes_dustribution(data):
    labels, counts = [], []
    for ngram, dimin_suffixes in sorted(data.items()):
        labels.append(ngram)
        counts.append(len(dimin_suffixes))

    df_viz = pd.DataFrame({'last letters': labels, 'suffix count': counts})
    sns.set_color_codes("pastel")
    sns.barplot(x='suffix count', y='last letters', data=df_viz, color='g')
    sns.despine(left=True, bottom=True)
    plt.show()



if __name__ == '__main__':
    PATH_TO_SAMPLES = '../data/train.tsv'

    df = read_samples(PATH_TO_SAMPLES, columns=['Name', 'Diminutive'])
    dist = get_diminutive_suffixes(df)
    simplify_suffixes(dist)
    plot_top_suffixes(dist)
    plot_suffixes_dustribution(dist)

