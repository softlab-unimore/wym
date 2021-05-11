import numpy as np

class Tokens_divider():
    def __init__(self, cols, word_vectors):
        self.word_vectors = word_vectors
        self.cols = cols
        self.all_cols = ['left_' + col for col in cols] + ['right_' + col for col in cols]

    def generate_most_similar_pairs(self, words_to_pair, given_words, threshold=0.15):
        if len(given_words) == 0 or len(words_to_pair) == 0:
            return {w: [] for w in words_to_pair}
        res = {}
        give_words = set(given_words)
        for word in words_to_pair:
            w = self.word_vectors.distances(word, given_words)
            index = [x <= threshold for x in w]
            w = w[index]
            tmp_words = np.array(given_words)[index]
            index = np.argsort(w)
            res[word] = np.stack([tmp_words[index], w[index]], axis=1)
        return res

    def assign_words_pairs(self, words_match, common_words, non_common):
        for col, words in words_match.items():
            for word, candidates in words.items():
                if len(candidates) > 0:
                    common_words[col.replace('left_', '').replace('right_', '')] += [(word, cand[0]) for cand in
                                                                                     candidates]
                else:
                    non_common[col].append(word)

    def compute_tokens_division(self, el_words):
        words_matches_lr = {}
        for col in self.cols:
            words_matches_lr['left_' + col] = self.generate_most_similar_pairs(el_words['left_' + col],
                                                                               el_words['right_' + col])
        common_words = {col: [] for col in self.cols}
        non_common_words = {col: [] for col in self.all_cols}
        self.assign_words_pairs(words_matches_lr, common_words, non_common_words)

        words_matches_rl = {}
        for col in self.cols:
            rwords = list(set(el_words['right_' + col]) - set([w[1] for w in common_words[col]]))
            lwords = list(set(el_words['left_' + col]) - set([w[0] for w in common_words[col]]))
            words_matches_rl['right_' + col] = self.generate_most_similar_pairs(rwords, lwords)
        self.assign_words_pairs(words_matches_rl, common_words, non_common_words)
        return common_words, non_common_words

    def tokens_in_vocab_division(self, item):
        index, el = item
        label = el['label']
        el_words = {}
        for col in self.cols:
            if el.isnull()[['left_' + col, 'right_' + col]].any():  # both description must be non Nan
                el_words['left_' + col] = []
                el_words['right_' + col] = []
            else:
                el_words['left_' + col] = str(el['left_' + col]).split()
                el_words['right_' + col] = str(el['right_' + col]).split()
        common, non_common = self.compute_tokens_division(el_words)
        additional_data = {'id': index, 'label': el.label}
        return self.to_list_of_dict(common, non_common, index, label)

    def to_list_of_dict(self, common_words, non_common_words, index, label):
        tuples_common = []
        tmp_dict = {}
        tmp_dict.update(id=index, label=label)
        for col in self.cols:
            tmp_dict.update(attribute=col)
            for pair in common_words[col]:
                tmp_dict.update(left_word=pair[0], right_word=pair[1])
                tuples_common.append(tmp_dict.copy())

        tuples_non_common = []
        tmp_dict = {}
        tmp_dict.update(id=index, label=label)
        for col in self.cols:
            tmp_dict.update(attribute=col)
            tmp_dict.update(side='left')
            prefix = 'left_'
            for word in non_common_words[prefix + col]:
                tmp_dict.update(word=word)
                tuples_non_common.append(tmp_dict.copy())

            tmp_dict.update(side='right')
            prefix = 'right_'
            for word in non_common_words[prefix + col]:
                tmp_dict.update(word=word)
                tuples_non_common.append(tmp_dict.copy())
        return tuples_common, tuples_non_common
