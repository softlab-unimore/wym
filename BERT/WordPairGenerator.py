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

    def __init__(self, words, embeddings, use_schema=False, **kwargs):
        super().__init__(**kwargs)
        self.words = words
        self.embeddings = embeddings
        self.use_schema = use_schema

    def process_df(self, df):
        word_dict = {'left_word': [], 'right_word': [], 'attribute': [], 'label': [],
                     'id': [], 'cos_sim': []}
        embedding_list = []
        for i in tqdm(range(df.shape[0])):
            gc.collect()
            torch.cuda.empty_cache()
            tmp_word, tmp_emb = self.generate_word_embedding_pairs(df.iloc[[i]])
            for key in tmp_word.keys():
                word_dict[key].append(tmp_word[key])
            embedding_list.append(tmp_emb)
        word_dic = {}
        for key in word_dict.keys():
            word_dict[key] = np.concatenate(word_dict[key])
        return word_dict, torch.cat(embedding_list)

    def generate_word_embedding_pairs(self, el, use_schema=True):
        words_pairs_dict, emb_pairs = self.embedding_pairs(el)
        total_word_pairs = {'left_word': [], 'right_word': [], 'attribute': [], 'cos_sim': []}
        total_emb_pairs = []
        for key in words_pairs_dict.keys():
            total_word_pairs[key].append(np.array(words_pairs_dict[key]))
        total_word_pairs['attribute'].append(['mixed'] * len(words_pairs_dict['right_word']))
        total_emb_pairs.append(emb_pairs)

        for key in total_word_pairs.keys():
            total_word_pairs[key] = np.hstack(total_word_pairs[key])
        n_pairs = len(total_word_pairs['left_word'])
        total_word_pairs['label'] = [el.label.values[0]] * n_pairs
        total_word_pairs['id'] = [el.id.values[0]] * n_pairs
        total_emb_pairs = torch.cat(total_emb_pairs)
        return total_word_pairs, total_emb_pairs

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
    def most_similar_pairs(sim_mat, threshold=.9):
        row_el = np.arange(sim_mat.shape[0])
        col_el = np.arange(sim_mat.shape[1])
        pairs = np.array((sim_mat > threshold).nonzero()).reshape([-1, 2])

        row_unpaired, col_unpaired = WordPairGenerator.get_not_paired(pairs, row_el, col_el)

        if len(row_unpaired) > 0 and len(col_unpaired) > 0:
            # Paired elements under the threshold of similarity --> max 1 pair per word.
            remaining_sim = sim_mat[row_unpaired][:, col_unpaired]
            row_max_mask = np.array(remaining_sim == remaining_sim.max(0))
            col_max_mask = np.array(remaining_sim.T == remaining_sim.max(1)).T
            if np.any(row_max_mask & col_max_mask):
                new_stable_pairs = np.array((row_max_mask & col_max_mask).nonzero())[:, [1, 0]]
                pairs = np.concatenate([pairs, new_stable_pairs])

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
        return pairs, sim


    def get_descriptions_to_compare(self, el):
        el_words = {}
        for col in self.cols:
            el_words[self.lp + col] = str(el[self.lp + col].values[0]).split()
            el_words[self.rp + col] = str(el[self.rp + col].values[0]).split()
            """
            if el.isnull()[[self.lp + col, self.rp + col]].values.any():  # both description must be non Nan
                el_words[self.lp + col] = []
                el_words[self.rp + col] = []
            else:
                el_words[self.lp + col] = str(el[self.lp + col].values[0]).split()
                el_words[self.rp + col] = str(el[self.rp + col].values[0]).split()
                
            """
        return el_words

    def generate_pairs(self, words_l, words_r, emb_l, emb_r):
        words_l, words_r = np.array(words_l + ['[UNP]']), np.array(words_r + ['[UNP]'])
        sim_mat = WordPairGenerator.cos_sim_set(emb_l.cpu(), emb_r.cpu())
        pairs, sim = WordPairGenerator.most_similar_pairs(sim_mat.cpu())
        emb_l = torch.cat([emb_l, torch.zeros_like(emb_l[[0]])], 0)
        emb_r = torch.cat([emb_r, torch.zeros_like(emb_r[[0]])], 0)
        word_pair = {'left_word': words_l[pairs[:, 0]].reshape([-1]), 'right_word': words_r[pairs[:, 1]].reshape([-1]),
                     'cos_sim': sim}
        return word_pair, torch.stack([emb_l[pairs[:, 0]], emb_r[pairs[:, 1]]]).permute(1, 0, 2)

    def embedding_pairs(self, el):
        emb1 = self.embeddings['table_A'][el.left_id.values[0]]
        emb2 = self.embeddings['table_B'][el.right_id.values[0]]
        words1 = self.words['table_A'][el.left_id.values[0]]
        words2 = self.words['table_B'][el.right_id.values[0]]
        if self.use_schema:
            el_words = self.get_descriptions_to_compare(el)
            word_pair = {'left_word': [], 'right_word': [], 'cos_sim': []}
            emb_pair = []
            for col in self.cols:
                indexes_l = [words1.index(x) for x in el_words[self.lp + col]]
                new_words_l = np.array(words1)[indexes_l]
                new_emb_l = emb1[indexes_l]
                indexes_r = [words2.index(x) for x in el_words[self.rp + col]]
                new_words_r = np.array(words2)[indexes_r]
                new_emb_r = emb2[indexes_r]
                tmp_word, tmp_emb = self.generate_pairs(new_words_l, new_words_r, new_emb_l, new_emb_r)
                for key in tmp_word.keys():
                    word_pair[key].append(tmp_word[key])
                emb_pair.append(tmp_emb)
            return word_pair, torch.cat(emb_pair)
        else:
            return self.generate_pairs(words1, words2, emb1, emb2)
