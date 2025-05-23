from collections import defaultdict, Counter
from random import choice, random

from utils.dim_io import read_samples


class DiminutiveGenerator:
    _KA_ENDING = 'ка'
    _RU_VOWELS = 'аеиоуыэюя'
    _LAST_LETTER = _RU_VOWELS + 'ь'
    _DIM_ENDING = '$'
    _START = '~'

    LANGUAGE_DEFAULT_PROB = 0.0001
    DIMINUTIVE_DEFAULT_PROB = 0.0001

    def __init__(self, ngram=2):
        if ngram < 2:
            raise Exception('Ngram parameter should be greater or equal 2 characters!')

        self.ngram = ngram

        self.lang_model = defaultdict(Counter)
        self.lang_endings_model = defaultdict(Counter)
        self.lang_endings_context = defaultdict(Counter)
        self.diminutive_transits = defaultdict(Counter)

    @staticmethod
    def __choose_letter(dist):
        x = random()
        for c, v in dist:
            x = x - v
            if x <= 0:
                return c

    @staticmethod
    def __normalize(counter):
        total = float(sum(counter.values()))
        return [(c, cnt / total) for c, cnt in counter.items()]

    def __normalize_transits(self, history, counter):
        def get_prob_denot(char):
            return self.lang_model[history][char]

        return [(c, (cnt / get_prob_denot(c[0])))
                for c, cnt in sorted(counter.items(), key=lambda x: x[0][-1])]

    def __train_lm(self, names):
        print('Collecting of letters\' probabilities in language model...')
        for real_name in names:
            real_name = f'{real_name.lower()}$'
            n_chars = self._START * self.ngram
            for char in real_name:
                self.lang_model[n_chars][char] += 1
                n_chars = n_chars[1:] + char

    def __train_diminutive_model(self, names, diminutives):
        print('Collecting of letters\' probabilities in diminutive model...')
        for real_name, diminutive in zip(names, diminutives):
            real_name, diminutive = real_name.lower(), f'{diminutive.lower()}$'
            stay_within_name = True
            n_chars = self._START * self.ngram
            max_len = max(len(real_name), len(diminutive))
            for i in range(max_len):
                if i < len(real_name) and stay_within_name:
                    ch, dim_ch = real_name[i], diminutive[i]
                    if ch != dim_ch:
                        stay_within_name = False
                        self.diminutive_transits[n_chars][(ch, dim_ch)] += 1
                        next_char = diminutive[i]
                    else:
                        next_char = real_name[i]
                    n_chars = n_chars[1:] + next_char
                else:
                    if i == len(real_name) and real_name.endswith(n_chars):
                        ch, dim_ch = '$', diminutive[i]
                        self.diminutive_transits[n_chars][(ch, dim_ch)] += 1
                    elif i < len(diminutive):
                        self.lang_endings_model[n_chars][diminutive[i]] += 1
                        self.lang_endings_context[n_chars[1:]][diminutive[i]] += 1
                    else:
                        break
                    n_chars = n_chars[1:] + diminutive[i]

    def fit(self, path_to_sample_file):
        print('Get data from the file...')
        df = read_samples(path_to_sample_file, columns=['Name', 'Diminutive'])

        # collect language model
        self.__train_lm(df.Name)

        # collect diminutive model
        self.__train_diminutive_model(df.Name, df.Diminutive)

        # normalize models
        self.lang_endings_model = {hist: self.__normalize(chars)
                                   for hist, chars in self.lang_endings_model.items()}
        self.lang_endings_context = {hist: self.__normalize(chars)
                                     for hist, chars in self.lang_endings_context.items()}
        self.diminutive_transits = {hist: self.__normalize_transits(hist, chars)
                                    for hist, chars in self.diminutive_transits.items()}
        self.lang_model = {hist: self.__normalize(chars) for hist, chars in self.lang_model.items()}

        return self

    def __get_lm_prob(self, hist, char):
        if hist not in self.lang_model:
            return self.LANGUAGE_DEFAULT_PROB
        else:
            for c, v in self.lang_model[hist]:
                if c == char:
                    return v
            return self.LANGUAGE_DEFAULT_PROB

    def __find_max_transition(self, word):
        # find the max prob Lang(history, char) * Transit(history, char) and extremal arguments

        max_hist = None
        letter, index = '', 0
        prob = self.DIMINUTIVE_DEFAULT_PROB
        start = len(word) // 2 + self.ngram

        for i in range(start, len(word)):
            ch, ngram_hist = word[i], word[i - self.ngram:i]
            if ngram_hist not in self.diminutive_transits:
                continue
            lm_prob = self.__get_lm_prob(ngram_hist, ch)
            for t, p in self.diminutive_transits[ngram_hist]:
                cond_prob = p * lm_prob
                if t[0] == ch and cond_prob >= prob:
                    prob, index, letter = cond_prob, i, t[0]
                    max_hist = self.diminutive_transits[ngram_hist]
        return index, letter, max_hist, prob

    def __find_default_transition(self, word):
        index = len(word) - (0 if word[-1] not in self._LAST_LETTER else 2 if word[-2:] in ['ха'] else 1)
        letter = 'х' if word[-2:] in ['ха'] else '$' if word[-1] not in self._LAST_LETTER else word[-1]

        ngram = self.ngram - 1
        histories_by_last_ch = [
            (h, _) for h, _ in self.diminutive_transits.items() if h.endswith(word[index - ngram:index])
        ]
        if histories_by_last_ch:
            hists_by_last_ch = self.__select_hists_by_char(histories_by_last_ch, letter)
            if hists_by_last_ch:
                histories_by_last_ch = hists_by_last_ch
            return index, letter, choice(histories_by_last_ch)[-1]
        else:
            return None

    def __generate_letter(self, history):
        if history not in self.lang_endings_model:
            dist = self.lang_endings_context.get(history[1:], None)
        else:
            dist = self.lang_endings_model[history]

        if not dist:
            return self._KA_ENDING

        return self.__choose_letter(dist)

    def __generate_diminutive_tail(self, history):
        while True:
            c = self.__generate_letter(history)
            if c == self._DIM_ENDING:
                break

            yield c
            if c == self._KA_ENDING:
                break

            history = history[-self.ngram + 1:] + c

    @classmethod
    def normalize_k_suffix(cls, word):
        if len(word) > 3 and word.endswith(cls._KA_ENDING):
            if word[-3] in 'йь':
                return word[:-3] + 'я'
            elif word[-3] not in cls._RU_VOWELS:
                return word[:-2] + word[-1]
        return word

    @staticmethod
    def __select_hists_by_char(hists, char):
        candidates = []
        for h, d in hists:
            curr_transits = [(k, _) for k, _ in d if k[0] == char]
            if curr_transits:
                candidates += [(h, curr_transits)]

        return candidates

    def generate_diminutive(self, word, print_euristic_flag=False):
        
        if print_euristic_flag:
            used_euristic = False

        # check if word has 'ка' ending and normalize name
        word = self.normalize_k_suffix(word)

        # fill name with ngram start
        n_chars = self._START * self.ngram
        word = n_chars + word.lower()

        # find transition with max probability
        index, letter, max_hist, prob = self.__find_max_transition(word)

        # process last name's symbols with default probability
        if prob <= self.DIMINUTIVE_DEFAULT_PROB:
            default_params = self.__find_default_transition(word)
            if not default_params:
                return word[self.ngram:].capitalize()

            index, letter, max_hist = default_params
            if print_euristic_flag:
                used_euristic = True

        # select transits with first character which equal the letter of a name
        max_hist_for_letter = [(tr, _) for tr, _ in max_hist if tr[0] == letter]
        if max_hist_for_letter:
            max_hist = max_hist_for_letter

        if not max_hist:
            return word[self.ngram:].capitalize()

        # generate a tail of the diminutive (to 'a' character)
        dim_letter = self.__choose_letter(max_hist)
        first_dim_letter = max(max_hist, key=lambda x: x[-1])[0][-1] if not dim_letter else dim_letter[-1]
        result = word[:index] + first_dim_letter
        tail = ''.join(self.__generate_diminutive_tail(result[-self.ngram:]))

        diminutive_name = result[self.ngram:].capitalize() + tail
        return diminutive_name if not print_euristic_flag else (diminutive_name, used_euristic)


if __name__ == '__main__':
    CORPUS_TRAIN = '../data/train.tsv'
    CORPUS_TEST = '../data/test.tsv'

    gen = DiminutiveGenerator(ngram=3)
    gen.fit(CORPUS_TRAIN)

    data = read_samples(CORPUS_TEST, columns=['Name'])
    for name in data.Name:
        print(gen.generate_diminutive(name))
