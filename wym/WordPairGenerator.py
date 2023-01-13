import gc
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from tqdm.autonotebook import tqdm
from .StableMarriage import gale_shapley


class EMFeatures:
    def __init__(self, df, exclude_attrs=('id', 'left_id', 'right_id', 'label'), device='cuda', n_proc=1):
        self.n_proc = n_proc
        self.cols = [x[5:] for x in df.columns if x not in exclude_attrs and x.startswith('left_')]
        self.lp = 'left_'
        self.rp = 'right_'
        self.all_cols = [self.lp + col for col in self.cols] + [self.rp + col for col in self.cols]
        self.device = device


def parallelize_dataframe(param_list, func=None, n_cores=4):
    with Pool(n_cores) as pool:
        df = pool.starmap(func, param_list)
        pool.close()
        pool.join()
    return df


class WordPairGenerator(EMFeatures):
    word_pair_empty = {'left_word': [], 'right_word': [], 'cos_sim': [], 'left_attribute': [],
                       'right_attribute': []}
    unpair_threshold = 0.6
    cross_attr_threshold = .65
    duplicate_threshold = .75 #.85 #1.1  # .75 #TODO adjust duplicate threshold

    def __init__(self, words=None, embeddings=None, words_divided=None, use_schema=True, sentence_embedding_dict=None,
                 unpair_threshold=None, cross_attr_threshold=None, duplicate_threshold=None,
                 verbose=False, size=768,
                 **kwargs):
        super().__init__(**kwargs)
        self.words = words
        self.embeddings = embeddings
        self.zero_emb = torch.zeros(1, size)
        self.use_schema = use_schema
        self.sentence_embedding_dict = sentence_embedding_dict
        self.unpair_threshold = unpair_threshold if unpair_threshold is not None else WordPairGenerator.unpair_threshold
        self.cross_attr_threshold = cross_attr_threshold if cross_attr_threshold is not None else WordPairGenerator.cross_attr_threshold
        self.duplicate_threshold = duplicate_threshold if duplicate_threshold is not None else WordPairGenerator.duplicate_threshold
        self.words_divided = words_divided
        self.verbose = verbose

    def get_word_pairs(self, df, data_dict):
        word_dict_list = []
        embedding_list = []
        if self.sentence_embedding_dict:
            sentence_embedding_list = []
            sent_emb_l, sent_emb_r = data_dict['left_sentence_emb'], data_dict['right_sentence_emb']
        else:
            sent_emb_l, sent_emb_r = [None] * df.shape[0], [None] * df.shape[0]
        if 'id' not in df.columns:
            df['id'] = df.index
        if self.verbose:
            print('generating word_pairs')
            to_cycle = tqdm(range(df.shape[0]))
        else:
            to_cycle = range(df.shape[0])

        gc.collect()
        torch.cuda.empty_cache()
        for i, words1, emb1, left_words_map, sent1, words2, emb2, right_words_map, sent2 in zip(
                to_cycle,
                data_dict['left_words'], data_dict['left_emb'], data_dict['left_word_map'], sent_emb_l,
                data_dict['right_words'], data_dict['right_emb'], data_dict['right_word_map'], sent_emb_r):
            if i % 1000 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            el = df.iloc[[i]]

            tmp_res = self.pairing_core_logic(el, emb1, emb2, words1, words2, left_words_map, right_words_map, sent1,
                                              sent2)

            # display(tmp_res)
            if self.sentence_embedding_dict:
                tmp_word, tmp_emb, tmp_sent_emb = tmp_res
                sentence_embedding_list.append(tmp_sent_emb)
            else:
                tmp_word, tmp_emb = tmp_res

            n_pairs = len(tmp_word['left_word'])
            if 'label' in el.columns:
                tmp_word['label'] = [el.label.values[0]] * n_pairs
            else:
                tmp_word['label'] = [0] * n_pairs
            tmp_word['id'] = [el.id.values[0]] * n_pairs
            word_dict_list.append(tmp_word)
            embedding_list.append(tmp_emb)


        keys = word_dict_list[0].keys()
        ret_dict = {key: np.concatenate([x[key] for x in word_dict_list]) for key in keys}
        # display(ret_dict)
        if self.sentence_embedding_dict is not None:
            return ret_dict, torch.cat(embedding_list), torch.cat(sentence_embedding_list)
        else:
            return ret_dict, torch.cat(embedding_list)

    # def get_word_pairs_parallel(self, df, data_dict):
    #     word_dict_list = []
    #     embedding_list = []
    #     if self.sentence_embedding_dict:
    #         sentence_embedding_list = []
    #         sent_emb_l, sent_emb_r = data_dict['left_sentence_emb'], data_dict['right_sentence_emb']
    #     else:
    #         sent_emb_l, sent_emb_r = [None] * df.shape[0], [None] * df.shape[0]
    #     if 'id' not in df.columns:
    #         df['id'] = df.index
    #     if self.verbose:
    #         print('generating word_pairs')
    #         to_cycle = range(df.shape[0])
    #     else:
    #         to_cycle = range(df.shape[0])
    #
    #     def do(i, words1, emb1, left_words_map, sent1, words2, emb2, right_words_map, sent2, df):
    #         el = df.iloc[[i]]
    #         return self.pairing_core_logic(el, emb1, emb2, words1, words2, left_words_map, right_words_map, sent1,
    #                                           sent2)
    #
    #     res_list = parallelize_dataframe(zip(
    #             to_cycle,
    #             data_dict['left_words'], data_dict['left_emb'], data_dict['left_word_map'], sent_emb_l,
    #             data_dict['right_words'], data_dict['right_emb'], data_dict['right_word_map'], sent_emb_r), partial(do,df=df) )
    #
    #     for i, tmp_res in enumerate(res_list):
    #         el = df.iloc[[i]]
    #         if self.sentence_embedding_dict:
    #             tmp_word, tmp_emb, tmp_sent_emb = tmp_res
    #             sentence_embedding_list.append(tmp_sent_emb)
    #         else:
    #             tmp_word, tmp_emb = tmp_res
    #
    #         n_pairs = len(tmp_word['left_word'])
    #         if 'label' in el.columns:
    #             tmp_word['label'] = [el.label.values[0]] * n_pairs
    #         else:
    #             tmp_word['label'] = [1] * n_pairs
    #         tmp_word['id'] = [el.id.values[0]] * n_pairs
    #         word_dict_list.append(tmp_word)
    #         embedding_list.append(tmp_emb)
    #
    #     keys = word_dict_list[0].keys()
    #     ret_dict = {key: np.concatenate([x[key] for x in word_dict_list]) for key in keys}
    #
    #     if self.sentence_embedding_dict is not None:
    #         return ret_dict, torch.cat(embedding_list), torch.cat(sentence_embedding_list)
    #     else:
    #         return ret_dict, torch.cat(embedding_list)

    def process_df(self, df):
        word_dict_list = []
        embedding_list = []
        if self.sentence_embedding_dict is not None:
            sentence_embedding_list = []
        to_cycle = tqdm(range(df.shape[0])) if self.verbose == True else range(df.shape[0])
        for i in to_cycle:
            if i % 2000 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            el = df.iloc[[i]]
            tmp_res = self.pairing_core_logic(el)

            if self.sentence_embedding_dict is not None:
                tmp_word, tmp_emb, tmp_sent_emb = tmp_res
                sentence_embedding_list.append(tmp_sent_emb)
            else:
                tmp_word, tmp_emb = tmp_res
            n_pairs = len(tmp_word['left_word'])
            tmp_word['label'] = [el.label.values[0]] * n_pairs
            tmp_word['id'] = [el.id.values[0]] * n_pairs
            word_dict_list.append(tmp_word)
            embedding_list.append(tmp_emb)

        keys = word_dict_list[0].keys()
        ret_dict = {key: np.concatenate([x[key] for x in word_dict_list]) for key in keys}
        # assert 2 in  ret_dict['id']
        if self.sentence_embedding_dict is not None:
            assert torch.cat(sentence_embedding_list).shape[0] == torch.cat(embedding_list).shape[
                0]  # TODO remove assert
            return ret_dict, torch.cat(embedding_list), torch.cat(sentence_embedding_list)
        else:
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
    def most_similar_pairs(sim_mat, duplicate_threshold=None, unpair_threshold=None):
        unpair_threshold = unpair_threshold if unpair_threshold is not None else WordPairGenerator.unpair_threshold
        duplicate_threshold = duplicate_threshold if duplicate_threshold is not None else WordPairGenerator.duplicate_threshold

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

    def generate_pairs(self, words_l, words_r, emb_l, emb_r, return_pairs=False, unpair_threshold=None,
                       duplicate_threshold=None):
        unpair_threshold = unpair_threshold if unpair_threshold is not None else self.unpair_threshold
        duplicate_threshold = duplicate_threshold if duplicate_threshold is not None else self.duplicate_threshold

        if len(words_r) == 0 and len(words_l) == 0:
            pairs, sim, ret_emb = [], [], []
            word_pair = {'left_word': [], 'right_word': [], 'cos_sim': sim}
        else:
            if len(words_l) == 0:
                pairs = np.array([[-1, x] for x in range(len(words_r))])
                sim = np.array([0] * len(words_r))
                emb_l = self.zero_emb.to(self.device)
            elif len(words_r) == 0:
                pairs = np.array([[x, -1] for x in range(len(words_l))])
                sim = np.array([0] * len(words_l))
                emb_r = self.zero_emb.to(self.device)
            else:
                sim_mat = WordPairGenerator.cos_sim_set(emb_l.cpu(), emb_r.cpu())
                pairs, sim = WordPairGenerator.most_similar_pairs(sim_mat.cpu(),
                                                                  duplicate_threshold=duplicate_threshold,
                                                                  unpair_threshold=unpair_threshold)
            words_l, words_r = np.concatenate([words_l, ['[UNP]']]), np.concatenate([words_r, ['[UNP]']])
            emb_l = torch.cat([emb_l.to(self.device), self.zero_emb.to(self.device)], 0).to(self.device)
            emb_r = torch.cat([emb_r.to(self.device), self.zero_emb.to(self.device)], 0).to(self.device)
            # print(f'wl: {words_l} \nwr:{words_r} \n {emb_l.shape} -- {emb_r.shape} \n {pairs},{sim}')

            word_pair = {'left_word': words_l[pairs[:, 0]].reshape([-1]),
                         'right_word': words_r[pairs[:, 1]].reshape([-1]),
                         'cos_sim': sim}
            ret_emb = torch.stack([emb_l[pairs[:, 0]], emb_r[pairs[:, 1]]]).permute(1, 0, 2)
        if return_pairs:
            return word_pair, ret_emb, pairs
        else:
            return word_pair, ret_emb

    def pairing_core_logic(self, el, emb1=None, emb2=None, words1=None, words2=None, left_words_map=None,
                           right_words_map=None, sent_emb_1=None, sent_emb_2=None):
        if emb1 is None or emb2 is None or words1 is None or words2 is None:
            emb1 = self.embeddings['table_A'][el.left_id.values[0]]
            emb2 = self.embeddings['table_B'][el.right_id.values[0]]
            if self.sentence_embedding_dict is not None:
                sent_emb_1 = self.sentence_embedding_dict['table_A'][el.left_id.values[0]]
                sent_emb_2 = self.sentence_embedding_dict['table_B'][el.right_id.values[0]]
            words1 = self.words['table_A'][el.left_id.values[0]]
            words2 = self.words['table_B'][el.right_id.values[0]]
            left_words_map = self.words_divided['table_A'][el.left_id.values[0]]
            right_words_map = self.words_divided['table_B'][el.right_id.values[0]]

        if self.use_schema:
            # assert len(words1) == np.sum([len(x) for x in left_words.values() if x != ['']]), [words1, left_words]
            # assert len(words2) == np.sum([len(x) for x in right_words_map.values() if x != ['']]), [words2, right_words_map]
            # assert words2[0] == list(right_words_map.values())[0][0], [words2[0], list(right_words_map.values())]

            unpaired_words = deepcopy(WordPairGenerator.word_pair_empty)
            unpaired_words.update(left_pos=[], right_pos=[])
            unpaired_emb = {'left': [], 'right': []}
            word_pair = deepcopy(WordPairGenerator.word_pair_empty)
            emb_pair = []
            start_pos = {'left': 0, 'right': 0}
            tmp_words = {}
            tmp_emb = {}
            for col in left_words_map.keys():
                turn_start = deepcopy(start_pos)
                for side in ['left', 'right']:
                    start = start_pos[side]
                    words_list, word_map, emb, sent_emb = (
                        words1, left_words_map, emb1, sent_emb_1) if side == 'left' else (
                        words2, right_words_map, emb2, sent_emb_2)
                    indexes = [words_list[start:].index(x) for x in word_map[col]]
                    if len(indexes) > 0:
                        indexes = np.array(indexes) + start
                        tmp_words[side] = np.array(words_list)[indexes]
                        tmp_emb[side] = emb[indexes]
                        start_pos[side] += len(word_map[col])
                    else:
                        tmp_words[side] = []
                        tmp_emb[side] = self.zero_emb
                # assert len(tmp_words['left'])>0 or len(tmp_words['right'])
                tmp_word_pairs, tmp_emb_pairs, pairs = self.generate_pairs(tmp_words['left'], tmp_words['right'],
                                                                           tmp_emb['left'], tmp_emb['right'],
                                                                           return_pairs=True,
                                                                           duplicate_threshold=1.1)
                if len(tmp_emb_pairs) > 0:
                    paired_idx = []
                    for i, (l, r) in enumerate(pairs):
                        if l == -1 and r != -1:
                            unpaired_words['right_pos'].append(r + turn_start['right'])
                            unpaired_words['right_attribute'].append(col)
                        elif l != -1 and r == -1:
                            unpaired_words['left_pos'].append(l + turn_start['left'])
                            unpaired_words['left_attribute'].append(col)
                        else:
                            paired_idx.append(i)
                    emb_pair.append(tmp_emb_pairs[paired_idx].cpu())
                    for key in tmp_word_pairs.keys():
                        word_pair[key] = np.concatenate([word_pair[key], np.array(tmp_word_pairs[key])[paired_idx]])
                    for attr_key in ['left_attribute', 'right_attribute']:
                        word_pair[attr_key] = np.concatenate([word_pair[attr_key], [col] * len(paired_idx)])

            emb_pair = torch.cat(emb_pair) if len(emb_pair) > 0 else torch.tensor([])

            # Pair remaining UNPAIRED words crossing the attribute schema

            l_emb_unp = emb1[unpaired_words['left' + '_pos']] if len(
                unpaired_words['left' + '_pos']) > 0 else self.zero_emb
            r_emb_unp = emb2[unpaired_words['right' + '_pos']] if len(
                unpaired_words['right' + '_pos']) > 0 else self.zero_emb
            words_l, words_r = np.array(words1)[unpaired_words['left_pos']], np.array(words2)[
                unpaired_words['right_pos']]
            unpaired_words['left_word'] = words_l
            unpaired_words['right_word'] = words_r
            unpaired_emb['left'] = l_emb_unp
            unpaired_emb['right'] = r_emb_unp
            if len(l_emb_unp) > 0 and len(r_emb_unp) > 0:
                tmp_word_pairs, tmp_emb_pairs, pairs = self.generate_pairs(words_l, words_r, l_emb_unp, r_emb_unp,
                                                                           return_pairs=True,
                                                                           unpair_threshold=self.cross_attr_threshold,
                                                                           duplicate_threshold=1.1)
                new_unpaired_words = deepcopy(WordPairGenerator.word_pair_empty)
                new_unpaired_words.update(left_pos=[], right_pos=[])
                if len(tmp_emb_pairs) > 0:
                    paired_idx = []
                    for i, (l, r) in enumerate(pairs):
                        if l == -1 and r != -1:
                            new_unpaired_words['right_pos'].append(r)
                            new_unpaired_words['right_attribute'].append(unpaired_words['right_attribute'][r])
                        elif l != -1 and r == -1:
                            new_unpaired_words['left_pos'].append(l)
                            new_unpaired_words['left_attribute'].append(unpaired_words['left_attribute'][l])
                        else:
                            paired_idx.append(i)
                    emb_pair = torch.cat([emb_pair, tmp_emb_pairs[paired_idx].cpu()])
                    for key in tmp_word_pairs.keys():
                        word_pair[key] = np.concatenate([word_pair[key], np.array(tmp_word_pairs[key])[paired_idx]])
                    for side in ['left', 'right']:
                        paired_elements = pairs[paired_idx][:, 0 if side == 'left' else 1]
                        tmp_attr = np.array(unpaired_words[side + '_attribute'])[paired_elements]
                        word_pair[side + '_attribute'] = np.concatenate([word_pair[side + '_attribute'], tmp_attr])

                for side in ['left', 'right']:
                    new_unpaired_words[side + '_word'] = unpaired_words[side + '_word'][
                        new_unpaired_words[side + '_pos']]
                    unpaired_emb[side] = unpaired_emb[side][new_unpaired_words[side + '_pos']]
                unpaired_words = new_unpaired_words

            # Pair remaining UNPAIRED words with all opposite words (including already paired)
            # This generates duplication
            # First pair unpaired of left with all words of right
            # Then pair unpaired of right with all words of left

            for all_side, unp_side in zip(['left', 'right'], ['right', 'left']):
                words_map = right_words_map if all_side == 'right' else left_words_map
                pos_to_attr_map = WordPairGenerator.get_attr_map(words_map)
                emb_unp = unpaired_emb[unp_side] if len(
                    unpaired_words[unp_side + '_word']) > 0 else self.zero_emb
                if all_side == 'right':
                    tmp_word_pairs, tmp_emb, pairs = self.generate_pairs(unpaired_words[unp_side + '_word'], words2,
                                                                         emb_unp, emb2, return_pairs=True,
                                                                         unpair_threshold=self.duplicate_threshold,
                                                                         duplicate_threshold=1.1)
                elif all_side == 'left':
                    tmp_word_pairs, tmp_emb, pairs = self.generate_pairs(words1, unpaired_words[unp_side + '_word'],
                                                                         emb1, emb_unp, return_pairs=True,
                                                                         unpair_threshold=self.duplicate_threshold,
                                                                         duplicate_threshold=1.1)
                side_mask = np.array(tmp_word_pairs[unp_side + '_word']) != '[UNP]'
                for key in tmp_word_pairs.keys():
                    # display('first',word_pair[key],'and ', np.array(tmp_word_pairs[key])[side_mask])
                    word_pair[key] = np.concatenate(
                        [word_pair[key], np.array(tmp_word_pairs[key])[side_mask].flatten()])
                    # display(word_pair[key], '***',np.concatenate([word_pair[key], np.array(tmp_word_pairs[key])[side_mask]]))
                if len(pairs) >0:
                    all_attr = [pos_to_attr_map[pos] for pos in pairs[side_mask][:, 1 if all_side == 'right' else 0]]
                    word_pair[all_side + '_attribute'] = np.concatenate([word_pair[all_side + '_attribute'], all_attr])
                    unp_attr = np.array(unpaired_words[unp_side + '_attribute'])[
                        pairs[side_mask][:, 1 if unp_side == 'right' else 0]]
                    word_pair[unp_side + '_attribute'] = np.concatenate([word_pair[unp_side + '_attribute'], unp_attr])
                    emb_pair = torch.cat([emb_pair.cpu(), tmp_emb[side_mask].cpu()])

                for key in word_pair.keys():  # TODO remove in production
                    assert len(word_pair[key]) == len(
                        word_pair['left_word']), f'{key} --> {len(word_pair[key])} != {len(word_pair["left_word"])}'

        else:
            word_pair, emb_pair, pairs = self.generate_pairs(words1, words2, emb1, emb2, return_pairs=True)
            for side, word_dict in zip(['left', 'right'], [left_words_map, right_words_map]):
                pos_to_attr_map = WordPairGenerator.get_attr_map(word_dict)
                all_attr = [pos_to_attr_map[pos] for pos in pairs[:, 1 if side == 'right' else 0]]
                word_pair[side + '_attribute'] = np.array(all_attr)

        if self.sentence_embedding_dict is not None:
            tmp_array = torch.reshape(torch.stack([sent_emb_1, sent_emb_2]), (1, 2, -1))
            sent_emb_pair = torch.tile(tmp_array, (emb_pair.shape[0], 1, 1))
            return word_pair, emb_pair, sent_emb_pair
        else:
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


from nltk.metrics.distance import jaro_winkler_similarity
class WordPairGeneratorEdit(WordPairGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_word_pairs(self, df, data_dict):
        word_dict_list = []
        embedding_list = []
        if self.sentence_embedding_dict:
            sentence_embedding_list = []
            sent_emb_l, sent_emb_r = data_dict['left_sentence_emb'], data_dict['right_sentence_emb']
        else:
            sent_emb_l, sent_emb_r = [None] * df.shape[0], [None] * df.shape[0]
        if 'id' not in df.columns:
            df['id'] = df.index
        if self.verbose:
            print('generating word_pairs')
            to_cycle = tqdm(range(df.shape[0]))
        else:
            to_cycle = range(df.shape[0])

        gc.collect()
        torch.cuda.empty_cache()
        for i, left_words_map, right_words_map in zip(
                to_cycle,
                data_dict['left_word_map'],
                data_dict['right_word_map']):
            if i % 1000 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            el = df.iloc[[i]]

            tmp_res = self.pairing_core_logic(el, left_words_map, right_words_map)
            if self.sentence_embedding_dict:
                tmp_word, tmp_emb, tmp_sent_emb = tmp_res
                sentence_embedding_list.append(tmp_sent_emb)
            else:
                tmp_word = tmp_res

            n_pairs = len(tmp_word['left_word'])
            if 'label' in el.columns:
                tmp_word['label'] = [el.label.values[0]] * n_pairs
            else:
                tmp_word['label'] = [1] * n_pairs
            tmp_word['id'] = [el.id.values[0]] * n_pairs
            word_dict_list.append(tmp_word)

        keys = word_dict_list[0].keys()
        ret_dict = {key: np.concatenate([x[key] for x in word_dict_list]) for key in keys}
        return ret_dict

    def pairing_core_logic(self, el, left_words_map=None, right_words_map=None):
        if left_words_map is None and right_words_map is None:
            left_words_map = self.words_divided['table_A'][el.left_id.values[0]]
            right_words_map = self.words_divided['table_B'][el.right_id.values[0]]
        words1 = [word for phrase, attr in left_words_map.items() for word in phrase]
        words2 = [word for phrase, attr in right_words_map.items() for word in phrase]
        if self.use_schema:
            # assert len(words1) == np.sum([len(x) for x in left_words.values() if x != ['']]), [words1, left_words]
            # assert len(words2) == np.sum([len(x) for x in right_words_map.values() if x != ['']]), [words2, right_words_map]
            # assert words2[0] == list(right_words_map.values())[0][0], [words2[0], list(right_words_map.values())]

            unpaired_words = deepcopy(WordPairGenerator.word_pair_empty)
            unpaired_words.update(left_pos=[], right_pos=[])
            unpaired_emb = {'left': [], 'right': []}
            word_pair_dict_of_list = deepcopy(WordPairGenerator.word_pair_empty)
            emb_pair = []
            tmp_emb = {}
            for col in left_words_map.keys():
                tmp_word_pairs, pairs = self.generate_pairs(left_words_map[col], right_words_map[col],
                                                            return_pairs=True,
                                                            duplicate_threshold=1.1)
                if len(tmp_word_pairs) > 0:
                    paired_idx = []
                    for i, (l, r) in enumerate(pairs):
                        if l == -1 and r != -1:
                            unpaired_words['right_word'].append(right_words_map[col][r])
                            unpaired_words['right_attribute'].append(col)
                        elif l != -1 and r == -1:
                            unpaired_words['left_word'].append(left_words_map[col][l])
                            unpaired_words['left_attribute'].append(col)
                        else:
                            paired_idx.append(i)

                    for key in tmp_word_pairs.keys():
                        word_pair_dict_of_list[key] = np.concatenate(
                            [word_pair_dict_of_list[key], np.array(tmp_word_pairs[key])[paired_idx]])
                    for attr_key in ['left_attribute', 'right_attribute']:
                        word_pair_dict_of_list[attr_key] = np.concatenate(
                            [word_pair_dict_of_list[attr_key], [col] * len(paired_idx)])

            # Pair remaining UNPAIRED words crossing the attribute schema

            words_l, words_r = np.array(unpaired_words['left_word']), np.array(unpaired_words['right_pos'])
            if len(words_l) > 0 and len(words_r) > 0:
                tmp_word_pairs, pairs = self.generate_pairs(words_l, words_r,
                                                            return_pairs=True,
                                                            unpair_threshold=self.cross_attr_threshold,
                                                            duplicate_threshold=1.1)
                new_unpaired_words = deepcopy(WordPairGenerator.word_pair_empty)
                if len(tmp_word_pairs) > 0:
                    paired_idx = []
                    for i, (l, r) in enumerate(pairs):
                        if l == -1 and r != -1:
                            new_unpaired_words['right_word'].append(unpaired_words['right_word'][r])
                            new_unpaired_words['right_attribute'].append(unpaired_words['right_attribute'][r])
                        elif l != -1 and r == -1:
                            new_unpaired_words['left_word'].append(unpaired_words['left_word'][r])
                            new_unpaired_words['left_attribute'].append(unpaired_words['left_attribute'][r])
                        else:
                            paired_idx.append(i)

                    for key in tmp_word_pairs.keys():
                        word_pair_dict_of_list[key] = np.concatenate(
                            [word_pair_dict_of_list[key], np.array(tmp_word_pairs[key])[paired_idx]])
                    for attr_key in ['left_attribute', 'right_attribute']:
                        word_pair_dict_of_list[attr_key] = np.concatenate(
                            [word_pair_dict_of_list[attr_key], [col] * len(paired_idx)])
                    for side in ['left', 'right']:
                        paired_elements = pairs[paired_idx][:, 0 if side == 'left' else 1]
                        tmp_attr = np.array(unpaired_words[side + '_attribute'])[paired_elements]
                        word_pair_dict_of_list[side + '_attribute'] = np.concatenate(
                            [word_pair_dict_of_list[side + '_attribute'], tmp_attr])

                unpaired_words = new_unpaired_words

            # Pair remaining UNPAIRED words with all opposite words (including already paired)
            # This generates duplication
            # First pair unpaired of left with all words of right
            # Then pair unpaired of right with all words of left

            for all_side, unp_side in zip(['left', 'right'], ['right', 'left']):
                words_map = right_words_map if all_side == 'right' else left_words_map
                pos_to_attr_map = WordPairGenerator.get_attr_map(words_map)
                if all_side == 'right':
                    tmp_word_pairs, pairs = self.generate_pairs(unpaired_words[unp_side + '_word'], words2,
                                                                return_pairs=True,
                                                                unpair_threshold=self.duplicate_threshold,
                                                                duplicate_threshold=self.duplicate_threshold)
                elif all_side == 'left':
                    tmp_word_pairs, pairs = self.generate_pairs(words1, unpaired_words[unp_side + '_word'],
                                                                return_pairs=True,
                                                                unpair_threshold=self.duplicate_threshold,
                                                                duplicate_threshold=self.duplicate_threshold)
                side_mask = np.array(tmp_word_pairs[unp_side + '_word']) != '[UNP]'
                for key in tmp_word_pairs.keys():
                    # display('first',word_pair[key],'and ', np.array(tmp_word_pairs[key])[side_mask])
                    word_pair_dict_of_list[key] = np.concatenate(
                        [word_pair_dict_of_list[key], np.array(tmp_word_pairs[key])[side_mask].flatten()])
                    # display(word_pair[key], '***',np.concatenate([word_pair[key], np.array(tmp_word_pairs[key])[side_mask]]))
                all_attr = [pos_to_attr_map[pos] for pos in pairs[side_mask][:, 1 if all_side == 'right' else 0]]
                word_pair_dict_of_list[all_side + '_attribute'] = np.concatenate(
                    [word_pair_dict_of_list[all_side + '_attribute'], all_attr])
                unp_attr = np.array(unpaired_words[unp_side + '_attribute'])[
                    pairs[side_mask][:, 1 if unp_side == 'right' else 0]]
                word_pair_dict_of_list[unp_side + '_attribute'] = np.concatenate(
                    [word_pair_dict_of_list[unp_side + '_attribute'], unp_attr])

                for key in word_pair_dict_of_list.keys():  # TODO remove in production
                    assert len(word_pair_dict_of_list[key]) == len(
                        word_pair_dict_of_list[
                            'left_word']), f'{key} --> {len(word_pair_dict_of_list[key])} != {len(word_pair_dict_of_list["left_word"])}'

        else:
            word_pair_dict_of_list, pairs = self.generate_pairs(words1, words2, return_pairs=True)
            for side, word_dict in zip(['left', 'right'], [left_words_map, right_words_map]):
                pos_to_attr_map = WordPairGenerator.get_attr_map(word_dict)
                all_attr = [pos_to_attr_map[pos] for pos in pairs[:, 1 if side == 'right' else 0]]
                word_pair_dict_of_list[side + '_attribute'] = np.array(all_attr)

        return word_pair_dict_of_list

    def generate_pairs(self, words_l, words_r, return_pairs=False, unpair_threshold=None,
                       duplicate_threshold=None):
        unpair_threshold = unpair_threshold if unpair_threshold is not None else self.unpair_threshold
        duplicate_threshold = duplicate_threshold if duplicate_threshold is not None else self.duplicate_threshold

        if len(words_r) == 0 and len(words_l) == 0:
            pairs, sim = [], []
            word_pair = {'left_word': [], 'right_word': [], 'cos_sim': sim}
        else:
            if len(words_l) == 0:
                pairs = np.array([[-1, x] for x in range(len(words_r))])
                sim = np.array([0] * len(words_r))
            elif len(words_r) == 0:
                pairs = np.array([[x, -1] for x in range(len(words_l))])
                sim = np.array([0] * len(words_l))
            else:
                sim_mat = WordPairGeneratorEdit.sim_set(words_l, words_r)
                pairs, sim = WordPairGenerator.most_similar_pairs(sim_mat,
                                                                  duplicate_threshold=duplicate_threshold,
                                                                  unpair_threshold=unpair_threshold)
            words_l, words_r = np.concatenate([words_l, ['[UNP]']]), np.concatenate([words_r, ['[UNP]']])

            word_pair = {'left_word': words_l[pairs[:, 0]].reshape([-1]),
                         'right_word': words_r[pairs[:, 1]].reshape([-1]),
                         'cos_sim': sim}
        if return_pairs:
            return word_pair, pairs
        else:
            return word_pair

    @staticmethod
    def sim_set(words_l, words_r, sim_func=jaro_winkler_similarity):
        lev_app = lambda s: sim_func(s[0], s[1])
        rep1 = len(words_l)
        rep2 = len(words_r)
        return pd.DataFrame({0: pd.Series(words_l).repeat(rep2).values, 1: words_r * rep1}).apply(lev_app,
                                                                                                  1).values.reshape(
            [rep1, -1])

    def process_df(self, df):
        word_dict_list = []
        to_cycle = tqdm(range(df.shape[0])) if self.verbose == True else range(df.shape[0])
        for i in to_cycle:
            if i % 2000 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            el = df.iloc[[i]]
            tmp_res = self.pairing_core_logic(el)
            tmp_word = tmp_res
            n_pairs = len(tmp_word['left_word'])
            tmp_word['label'] = [el.label.values[0]] * n_pairs
            tmp_word['id'] = [el.id.values[0]] * n_pairs
            word_dict_list.append(tmp_word)

        keys = word_dict_list[0].keys()
        ret_dict = {key: np.concatenate([x[key] for x in word_dict_list]) for key in keys}

        return ret_dict
