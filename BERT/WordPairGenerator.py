import gc
from copy import deepcopy

import numpy as np
import torch
from tqdm.notebook import tqdm

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
    word_pair_empty = {'left_word': [], 'right_word': [], 'cos_sim': [], 'left_attribute': [],
                       'right_attribute': []}
    unpair_threshold = 0.45
    cross_attr_threshold = .55
    zero_emb = torch.zeros(1, 768)

    def __init__(self, words=None, embeddings=None, words_divided=None, use_schema=True, unpair_threshold=None,
                     cross_attr_threshold=None,
                     duplicate_threshold=.75, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.words = words
        self.embeddings = embeddings
        self.use_schema = use_schema
        self.unpair_threshold = unpair_threshold if unpair_threshold is not None else WordPairGenerator.unpair_threshold
        self.cross_attr_threshold = cross_attr_threshold if cross_attr_threshold is not None else WordPairGenerator.cross_attr_threshold
        self.duplicate_threshold = duplicate_threshold
        self.words_divided = words_divided
        self.verbose = verbose
        self.unpair_threshold



    def get_word_pairs(self, df, data_dict):
        word_dict_list = []
        embedding_list = []
        if 'id' not in df.columns:
            df['id'] = df.index
        if self.verbose:
            print('generating word_pairs')
            to_cycle = tqdm(range(df.shape[0]))
        else:
            to_cycle = range(df.shape[0])
        for i, words1, emb1, left_words_map, words2, emb2, right_words_map in zip(
                to_cycle,
                data_dict['left_words'], data_dict['left_emb'], data_dict['left_word_map'],
                data_dict['right_words'], data_dict['right_emb'], data_dict['right_word_map']):
            if i % 2000 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            el = df.iloc[[i]]
            tmp_word, tmp_emb = self.embedding_pairs(el, emb1, emb2, words1, words2, left_words_map, right_words_map)
            n_pairs = len(tmp_word['left_word'])
            if 'label' in el.columns:
                tmp_word['label'] = [el.label.values[0]] * n_pairs
            else:
                tmp_word['label'] = [1] * n_pairs
            tmp_word['id'] = [el.id.values[0]] * n_pairs
            word_dict_list.append(tmp_word)
            embedding_list.append(tmp_emb)

        keys = word_dict_list[0].keys()
        ret_dict = {key: np.concatenate([x[key] for x in word_dict_list]) for key in keys}

        """def tmp_func(obj, self):
                    i, words1, emb1, left_words_map, words2, emb2, right_words_map = obj
                    gc.collect()
                    torch.cuda.empty_cache()
                    el = df.iloc[[i]]
                    tmp_word, tmp_emb = self.embedding_pairs(el,  emb1, emb2, words1, words2, left_words_map, right_words_map)
                    n_pairs = len(tmp_word['left_word'])
                    tmp_word['label'] = [el.label.values[0]] * n_pairs
                    tmp_word['id'] = [el.id.values[0]] * n_pairs
                    print(i)
                    return [tmp_word,tmp_emb]


                pool = Pool(n_proc)
                res = pool.map(tmp_func, zip(to_precess, [deepcopy(self)]*len(list(to_precess)))  ### Function and iterable here
                pool.close()
                pool.join()

                for tmp_word, tmp_emb in res:
                    word_dict_list.append(tmp_word)
                    embedding_list.append(tmp_emb)"""
        return ret_dict, torch.cat(embedding_list)

    def process_df(self, df):
        word_dict_list = []
        embedding_list = []
        to_cycle = tqdm(range(df.shape[0])) if self.verbose == True else range(df.shape[0])
        for i in to_cycle:
            if i % 2000 == 0:
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
    def most_similar_pairs(sim_mat, duplicate_threshold=.75, unpair_threshold=None):
        unpair_threshold = unpair_threshold if unpair_threshold is not None else WordPairGenerator.unpair_threshold

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

    def generate_pairs(self, words_l, words_r, emb_l, emb_r, return_pairs=False, unpair_threshold=None):
        unpair_threshold = unpair_threshold if unpair_threshold is not None else WordPairGenerator.unpair_threshold

        if len(words_r) == 0 and len(words_l) == 0:
            pairs = []
            sim = []
            word_pair = {'left_word': [],
                         'right_word': [],
                         'cos_sim': sim}
            ret_emb = []

        else:
            if len(words_l) == 0:
                pairs = np.array([[-1, x] for x in range(len(words_r))])
                sim = np.array([0] * len(words_r))
                emb_l = WordPairGenerator.zero_emb.to(self.device)
            elif len(words_r) == 0:
                pairs = np.array([[x, -1] for x in range(len(words_l))])
                sim = np.array([0] * len(words_l))
                emb_r = WordPairGenerator.zero_emb.to(self.device)
            else:
                sim_mat = WordPairGenerator.cos_sim_set(emb_l.cpu(), emb_r.cpu())
                pairs, sim = WordPairGenerator.most_similar_pairs(sim_mat.cpu(),
                                                                  duplicate_threshold=self.duplicate_threshold,
                                                                  unpair_threshold=unpair_threshold)
            words_l, words_r = np.concatenate([words_l, ['[UNP]']]), np.concatenate([words_r, ['[UNP]']])
            emb_l = torch.cat([emb_l.to(self.device), WordPairGenerator.zero_emb.to(self.device)], 0).to(self.device)
            emb_r = torch.cat([emb_r.to(self.device), WordPairGenerator.zero_emb.to(self.device)], 0).to(self.device)
            # print(f'wl: {words_l} \nwr:{words_r} \n {emb_l.shape} -- {emb_r.shape} \n {pairs},{sim}')

            word_pair = {'left_word': words_l[pairs[:, 0]].reshape([-1]), 'right_word': words_r[pairs[:, 1]].reshape([-1]),
                         'cos_sim': sim}
            ret_emb = torch.stack([emb_l[pairs[:, 0]], emb_r[pairs[:, 1]]]).permute(1, 0, 2)
        if return_pairs:
            return word_pair, ret_emb, pairs
        else:
            return word_pair, ret_emb



    def embedding_pairs(self, el, emb1=None, emb2=None, words1=None, words2=None, left_words_map=None,
                        right_words_map=None):
        if emb1 is None or emb2 is None or words1 is None or words2 is None:
            emb1 = self.embeddings['table_A'][el.left_id.values[0]]
            emb2 = self.embeddings['table_B'][el.right_id.values[0]]
            words1 = self.words['table_A'][el.left_id.values[0]]
            words2 = self.words['table_B'][el.right_id.values[0]]
            left_words_map = self.words_divided['table_A'][el.left_id.values[0]]
            right_words_map = self.words_divided['table_B'][el.right_id.values[0]]

        if self.use_schema:
            # assert len(words1) == np.sum([len(x) for x in left_words.values() if x != ['']]), [words1, left_words]
            # assert len(words2) == np.sum([len(x) for x in right_words_map.values() if x != ['']]), [words2, right_words_map]
            # assert words2[0] == list(right_words_map.values())[0][0], [words2[0], list(right_words_map.values())]
            unpaired_words = deepcopy(WordPairGenerator.word_pair_empty)
            unpaired_emb = {'left': [], 'right': []}
            word_pair = deepcopy(WordPairGenerator.word_pair_empty)
            emb_pair = []
            start_pos = {'left': 0, 'right': 0}
            tmp_words = {}
            tmp_emb = {}
            for col in left_words_map.keys():
                for side in ['left', 'right']:
                    start = start_pos[side]
                    words_list, word_map, emb = (words1, left_words_map, emb1) if side == 'left' else (
                    words2, right_words_map, emb2)
                    indexes = [words_list[start:].index(x) for x in word_map[col]]
                    if len(indexes) > 0:
                        indexes = np.array(indexes) + start
                        tmp_words[side] = np.array(words_list)[indexes]
                        tmp_emb[side] = emb[indexes]
                        start_pos[side] += len(word_map[col])
                    else:
                        tmp_words[side] = []
                        tmp_emb[side] = WordPairGenerator.zero_emb

                tmp_word_pairs, tmp_emb_pairs = self.generate_pairs(tmp_words['left'], tmp_words['right'],
                                                                    tmp_emb['left'], tmp_emb['right'])
                if len(tmp_emb_pairs)> 0:
                    paired_idx = []
                    for i, (w_l, w_r, emb) in enumerate(
                            zip(tmp_word_pairs['left_word'], tmp_word_pairs['right_word'], tmp_emb_pairs)):
                        if w_l == '[UNP]' and w_r != '[UNP]':
                            unpaired_words['right_word'].append(w_r)
                            unpaired_words['right_attribute'].append(col)
                            unpaired_emb['right'].append(emb[1])
                        elif w_r == '[UNP]' and w_l != '[UNP]':
                            unpaired_words['left_word'].append(w_l)
                            unpaired_words['left_attribute'].append(col)
                            unpaired_emb['left'].append(emb[0])
                        else:
                            paired_idx.append(i)
                    emb_pair.append(tmp_emb_pairs[paired_idx].cpu())
                    for key in tmp_word_pairs.keys():
                        word_pair[key] = np.concatenate([word_pair[key], np.array(tmp_word_pairs[key])[paired_idx]])
                    for attr_key in ['left_attribute', 'right_attribute']:
                        word_pair[attr_key] = np.concatenate([word_pair[attr_key], [col] * len(paired_idx)])

            emb_pair = torch.cat(emb_pair) if len(emb_pair) > 0 else torch.tensor([])

            # Pair remaining words crossing the attribute schema
            # First pair unpaired of left with all words of right
            # Then pair unpaired of right with all words of left

            for all_side, unp_side in zip(['left', 'right'], ['right', 'left']):
                words_map = right_words_map if all_side == 'right' else left_words_map
                pos_to_attr_map = WordPairGenerator.get_attr_map(words_map)
                emb_unp = torch.stack(unpaired_emb[unp_side]) if len(
                    unpaired_words[unp_side + '_word']) > 0 else WordPairGenerator.zero_emb
                if all_side == 'right':
                    tmp_word_pairs, tmp_emb, pairs = self.generate_pairs(unpaired_words[unp_side + '_word'], words2,
                                                                         emb_unp, emb2, unpair_threshold=WordPairGenerator.cross_attr_threshold,
                                                                         return_pairs=True)
                elif all_side == 'left':
                    tmp_word_pairs, tmp_emb, pairs = self.generate_pairs(words1, unpaired_words[unp_side + '_word'],
                                                                         emb1, emb_unp, unpair_threshold=WordPairGenerator.cross_attr_threshold,
                                                                         return_pairs=True)
                side_mask = np.array(tmp_word_pairs[unp_side + '_word']) != '[UNP]'
                for key in tmp_word_pairs.keys():
                    word_pair[key] = np.concatenate([word_pair[key], np.array(tmp_word_pairs[key])[side_mask]])
                    # display(word_pair[key], '***',np.concatenate([word_pair[key], np.array(tmp_word_pairs[key])[side_mask]]))
                all_attr = [pos_to_attr_map[pos] for pos in pairs[side_mask][:, 1 if all_side == 'right' else 0]]
                word_pair[all_side + '_attribute'] = np.concatenate([word_pair[all_side + '_attribute'], all_attr])
                unp_attr = np.array(unpaired_words[unp_side + '_attribute'])[
                    pairs[side_mask][:, 1 if unp_side == 'right' else 0]]
                word_pair[unp_side + '_attribute'] = np.concatenate([word_pair[unp_side + '_attribute'], unp_attr])
                emb_pair = torch.cat([emb_pair.cpu(), tmp_emb[side_mask].cpu()])
                for key in word_pair.keys():
                    assert len(word_pair[key]) == len(
                        word_pair['left_word']), f'{key} --> {len(word_pair[key])} != {len(word_pair["left_word"])}'
            return word_pair, emb_pair
        else:
            word_pair, emb_pair, pairs = self.generate_pairs(words1, words2, emb1, emb2, return_pairs=True)
            for side, word_dict in zip(['left', 'right'], [left_words_map, right_words_map]):
                pos_to_attr_map = WordPairGenerator.get_attr_map(word_dict)
                all_attr = [pos_to_attr_map[pos] for pos in pairs[:, 1 if side == 'right' else 0]]
                word_pair[side + '_attribute'] = np.array(all_attr)
            return word_pair, emb_pair

    @staticmethod
    def map_word_to_attr(df, cols, prefix='', verbose=False):
        tmp_res = []

        if verbose:
            print('Mapping word to attr')
        to_cycle = tqdm(range(df.shape[0])) if verbose else range(df.shape[0])
        for i in to_cycle:
            el = df.iloc[[i]]
            el_words = {}
            for col in cols:
                if el[prefix + col].notna().values[0]:
                    el_words[col] = str(el[prefix + col].values[0]).split()
                else:
                    el_words[col] = []
            tmp_res.append(el_words.copy())
        return tmp_res
