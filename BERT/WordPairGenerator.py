import numpy as np
import torch
from tqdm.notebook import tqdm
import gc
from StableMarriage import gale_shapley


class EMFeatures:
    def __init__(self, df, exclude_attrs=['id', 'left_id', 'right_id', 'label'], device='cuda', n_proc=1):
        self.n_proc = n_proc
        self.cols = [x[5:] for x in df.columns if x not in exclude_attrs and x.startswith('left_')]
        self.lp = 'left_'
        self.rp = 'right_'
        self.all_cols = [self.lp + col for col in self.cols] + [self.rp + col for col in self.cols]
        self.device = device


class WordPairGenerator(EMFeatures):


    def __init__(self, words=None, embeddings=None, words_divided=None, use_schema=True, unpair_threshold=0.55,
                 duplicate_threshold=.75, **kwargs):
        super().__init__(**kwargs)
        self.words = words
        self.embeddings = embeddings
        self.use_schema = use_schema
        self.unpair_threshold = unpair_threshold
        self.duplicate_threshold = duplicate_threshold
        self.words_divided = words_divided
        if embeddings is not None:
            i=0
            while len(self.embeddings['table_A'][i]) == 0:
                i+=1
            self.zero_emb = torch.stack([torch.zeros_like(self.embeddings['table_A'][i][0])])
        self.word_pair_empty = {'left_word': [], 'right_word': [], 'cos_sim': [], 'left_attribute': [],
                                'right_attribute': []}



    def get_word_pairs(self, df, data_dict):
        word_dict_list = []
        embedding_list = []
        for i, emb1, emb2, words1, words2, left_words_map, right_words_map in tqdm(zip(range(df.shape[0]),
                            data_dict['left_words'], data_dict['left_emb'], data_dict['left_word_map'],
                            data_dict['right_words'], data_dict['right_emb'], data_dict['right_word_map'])):
            gc.collect()
            torch.cuda.empty_cache()
            el = df.iloc[[i]]
            tmp_word, tmp_emb = self.embedding_pairs(el,  emb1, emb2, words1, words2, left_words_map, right_words_map)
            n_pairs = len(tmp_word['left_word'])
            tmp_word['label'] = [el.label.values[0]] * n_pairs
            tmp_word['id'] = [el.id.values[0]] * n_pairs
            word_dict_list.append(tmp_word)
            embedding_list.append(tmp_emb)

        keys = word_dict_list[0].keys()
        ret_dict = {key: np.concatenate([x[key] for x in word_dict_list]) for key in keys}
        return ret_dict, torch.cat(embedding_list)

    def process_df(self, df):
        word_dict_list = []
        embedding_list = []
        for i in tqdm(range(df.shape[0])):
            if i% 500 ==0:
                gc.collect()
                torch.cuda.empty_cache()
            el = df.iloc[[i]]
            tmp_word, tmp_emb = self.embedding_pairs(el)
            n_pairs = len(tmp_word['left_word'])
            tmp_word['label'] = [el.label.values[0]] * n_pairs
            tmp_word['id'] = [el.id.values[0]] * n_pairs
            word_dict_list.append(tmp_word)
            embedding_list.append(tmp_emb)

        keys = word_dict_list[0].keys()
        ret_dict = {key: np.concatenate([x[key] for x in word_dict_list]) for key in keys}
        return ret_dict, torch.cat(embedding_list)


    @staticmethod
    def cos_sim_set(emb1, emb2):
        cos = torch.nn.CosineSimilarity(eps=1e-6)
        rep1 = emb1.shape[0]
        rep2 = emb2.shape[0]
        return cos(emb1.repeat_interleave(rep2, 0), emb2.repeat(rep1, 1)).reshape([rep1, -1])

    @staticmethod
    def stable_marriage(a, b, a_pref, b_pref):
        # https://en.wikipedia.org/wiki/Stable_marriage_problem
        a_pref_dict = {el: pref for el, pref in zip(a, a_pref)}
        b_pref_dict = {el: pref for el, pref in zip(b, b_pref)}
        if len(a) > len(b):
            return np.array(gale_shapley(A=b, B=a, A_pref=b_pref_dict, B_pref=a_pref_dict))[:, [1, 0]]
        return np.array(gale_shapley(A=a, B=b, A_pref=a_pref_dict, B_pref=b_pref_dict))

    @staticmethod
    def get_not_paired(pairs, a, b):
        if pairs.shape[0] == 0:
            return a, b
        unpaired_el = lambda x, y: np.setdiff1d(y, np.unique(x))
        l_unpaired = unpaired_el(pairs[:, 0], a)
        r_unpaired = unpaired_el(pairs[:, 1], b)
        return l_unpaired, r_unpaired

    @staticmethod
    def high_similar_pairs(sim_mat, duplicate_threshold=.85):
        row_el = np.arange(sim_mat.shape[0])
        col_el = np.arange(sim_mat.shape[1])
        pairs = np.array((sim_mat > duplicate_threshold).nonzero()).reshape([-1, 2])
        pairs = np.unique(pairs, axis=0)
        row_unpaired, col_unpaired = WordPairGenerator.get_not_paired(pairs, row_el, col_el)
        unpaired = []
        for x in row_unpaired:
            unpaired.append([x, -1])
        for x in col_unpaired:
            unpaired.append([-1, x])
        if len(unpaired) > 0:
            pairs = np.concatenate([pairs, np.array(unpaired)])
        sim = [sim_mat[r, c] if r != -1 and c != -1 else 0 for r, c in pairs]

        # Drop word pairs with low sim.
        sim = np.array(sim)
        return pairs, sim

    @staticmethod
    def most_similar_pairs(sim_mat, duplicate_threshold=.75, unpair_threshold=0.55):

        row_el = np.arange(sim_mat.shape[0])
        col_el = np.arange(sim_mat.shape[1])
        pairs = np.array((sim_mat > duplicate_threshold).nonzero()).reshape([-1, 2])
        row_unpaired, col_unpaired = WordPairGenerator.get_not_paired(pairs, row_el, col_el)
        if len(row_unpaired) > 0 and len(col_unpaired) > 0:
            # Not stable pair under the threshold. constraint to 1 pair per word
            remaining_sim = sim_mat[row_unpaired][:, col_unpaired]
            row_preferences = np.array(col_unpaired[np.argsort(-remaining_sim)]).reshape(-1, len(col_unpaired))
            col_preferences = np.array(row_unpaired[np.argsort(-remaining_sim.T)]).reshape(-1, len(row_unpaired))
            new_pairs = WordPairGenerator.stable_marriage(row_unpaired, col_unpaired, row_preferences,
                                                          col_preferences)
            pairs = np.concatenate([pairs, new_pairs])
        pairs = np.unique(pairs, axis=0)
        row_unpaired, col_unpaired = WordPairGenerator.get_not_paired(pairs, row_el, col_el)
        unpaired = []
        for x in row_unpaired:
            unpaired.append([x, -1])
        for x in col_unpaired:
            unpaired.append([-1, x])
        if len(unpaired) > 0:
            pairs = np.concatenate([pairs, np.array(unpaired)])
        sim = [sim_mat[r, c] if r != -1 and c != -1 else 0 for r, c in pairs]

        # Drop word pairs with low sim.
        sim = np.array(sim)
        to_drop = np.where((sim < unpair_threshold) & (pairs[:, 0] != -1) & (pairs[:, 1] != -1))[0]  # return index
        if to_drop.shape[0] > 0:
            unpaired = []
            for r, c in pairs[to_drop]:
                unpaired.append([r, -1])
                unpaired.append([-1, c])

            pairs = np.delete(pairs, to_drop, axis=0)
            assert len(pairs.shape) == len(np.array(unpaired).shape), f'{pairs}, {unpaired}, {to_drop}'
            pairs = np.concatenate([pairs, np.array(unpaired)])
            sim = [sim_mat[r, c] if r != -1 and c != -1 else 0 for r, c in pairs]

        return pairs, sim

    @staticmethod
    def get_attr_map(words_dict):
        attr_len = [len(x) if x != [''] else 0 for x in words_dict.values()]
        pos_to_attr_map = {}
        pos = 0
        for i, attr in enumerate(words_dict.keys()):
            for word_pos in range(pos, pos + attr_len[i]):
                pos_to_attr_map[word_pos] = attr
            pos += attr_len[i]
        pos_to_attr_map[-1] = '[UNP]'
        return pos_to_attr_map

    def get_descriptions_to_compare(self, el):
        left_el = self.table_A.iloc[[el.left_id.values[0]]]
        right_el = self.table_B.iloc[[el.right_id.values[0]]]
        el_words = {}
        for prefix, record in zip([self.lp, self.rp], [left_el, right_el]):
            for col in self.cols:
                if record[col].notna().values[0]:
                    el_words[prefix + col] = str(record[col].values[0]).split()
        return el_words

    def generate_pairs(self, words_l, words_r, emb_l, emb_r, return_pairs=False):

        if len(words_r) == 0 and len(words_l) == 0:
            pairs = []
            sim = []
        elif len(words_l) == 0:
            pairs = np.array([[-1, x] for x in range(len(words_r))])
            sim = np.array([0] * len(words_r))
        elif len(words_r) == 0:
            pairs = np.array([[x, -1] for x in range(len(words_l))])
            sim = np.array([0] * len(words_l))
        else:
            sim_mat = WordPairGenerator.cos_sim_set(emb_l.cpu(), emb_r.cpu())
            pairs, sim = WordPairGenerator.most_similar_pairs(sim_mat.cpu(),
                                                              duplicate_threshold=self.duplicate_threshold,
                                                              unpair_threshold=self.unpair_threshold)
        words_l, words_r = np.concatenate([words_l, ['[UNP]']]), np.concatenate([words_r, ['[UNP]']])
        emb_l = torch.cat([emb_l, torch.zeros_like(emb_l[[0]])], 0).to(self.device)
        emb_r = torch.cat([emb_r, torch.zeros_like(emb_r[[0]])], 0).to(self.device)
        # print(f'wl: {words_l} \nwr:{words_r} \n {emb_l.shape} -- {emb_r.shape} \n {pairs},{sim}')

        word_pair = {'left_word': words_l[pairs[:, 0]].reshape([-1]), 'right_word': words_r[pairs[:, 1]].reshape([-1]),
                     'cos_sim': sim}
        ret_emb = torch.stack([emb_l[pairs[:, 0]], emb_r[pairs[:, 1]]]).permute(1, 0, 2)
        if return_pairs:
            return word_pair, ret_emb, pairs
        else:
            return word_pair, ret_emb

    def embedding_pairs(self, el, emb1=None, emb2=None, words1=None, words2=None, left_words_map=None, right_words_map=None):
        if emb1 is None or emb2 is None or words1 is None or words2 is None:
            emb1 = self.embeddings['table_A'][el.left_id.values[0]]
            emb2 = self.embeddings['table_B'][el.right_id.values[0]]
            words1 = self.words['table_A'][el.left_id.values[0]]
            words2 = self.words['table_B'][el.right_id.values[0]]
            left_words_map = self.words_divided['table_A'][el.left_id.values[0]]
            right_words_map = self.words_divided['table_B'][el.right_id.values[0]]


        if self.use_schema:
            #assert len(words1) == np.sum([len(x) for x in left_words.values() if x != ['']]), [words1, left_words]
            #assert len(words2) == np.sum([len(x) for x in right_words_map.values() if x != ['']]), [words2, right_words_map]
            # assert words2[0] == list(right_words_map.values())[0][0], [words2[0], list(right_words_map.values())]

            word_pair, emb_pair, pairs = self.generate_pairs(words1, words2, emb1, emb2, return_pairs=True)
            for side, word_dict in zip(['left', 'right'], [left_words_map, right_words_map]):
                pos_to_attr_map = WordPairGenerator.get_attr_map(word_dict)
                all_attr = [pos_to_attr_map[pos] for pos in pairs[:, 1 if side == 'right' else 0]]
                word_pair[side + '_attribute'] = np.array(all_attr)

            return word_pair, emb_pair
        else:
            word_pair, emb = self.generate_pairs(words1, words2, emb1, emb2)
            word_pair['attribute'] = ['mixed'] * len(word_pair['left_word'])
            return word_pair, emb


    @staticmethod
    def map_word_to_attr(df, cols, prefix=''):
        tmp_res = []
        for i in tqdm(range(df.shape[0])):
            el = df.iloc[[i]]
            el_words = {}
            for col in cols:
                if el[prefix+col].notna().values[0]:
                    el_words[col] = str(el[prefix+col].values[0]).split()
                else:
                    el_words[col] = []
            tmp_res.append(el_words.copy())
        return tmp_res