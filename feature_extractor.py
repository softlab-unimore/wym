import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from Token_divider import Tokens_divider


class FeatureExtractorGeneral:
    def __init__(self, df, word_vectors,
                 exclude_attrs=['id', 'left_id', 'right_id', 'label'], n_proc=1):
        self.n_proc = n_proc
        self.word_vectors = word_vectors
        self.cols = [x[5:] for x in df.columns if x not in exclude_attrs and x.startswith('left_')]
        self.lp = 'left_'
        self.rp = 'right_'
        self.all_cols = [self.lp + col for col in self.cols] + [self.rp + col for col in self.cols]

    @staticmethod
    def compute_min_max_features(df, columns):
        res = []
        for col in columns:
            res.append(pd.Series(
                np.where(df[col + '_left'] < df[col + '_right'],
                         df[col + '_left'], df[col + '_right']), name=col + '_unpaired_min'))
            res.append(pd.Series(
                np.where(df[col + '_left'] > df[col + '_right'],
                         df[col + '_left'], df[col + '_right']), name=col + '_unpaired_max'))
        res = pd.concat(res, axis=1)
        res.index.name = 'id'
        return res

    @staticmethod
    def compute_derived_features(df, feature_names):
        for feature_name in feature_names:
            for x in ['_max', '_min', '']:
                if x in ['_max','_min']:
                    suffix = '_min' if x == '_max' else '_max'
                else:
                    suffix = ''
                df[feature_name + '_diff' + suffix] = df[feature_name + '_paired'] - df[
                    feature_name + '_unpaired' + x]
                df[feature_name + '_perc' + suffix] = df[feature_name + '_paired'] + 1e-9 / ( 1e-9 + df[feature_name + '_paired'] + df[
                    feature_name + '_unpaired' + x])
            # df[feature_name + '_diff' + '_2min'] = df[feature_name + '_paired'] - (df[feature_name + '_unpaired_min'] * 2)
            # df[feature_name + '_perc' + '_2min'] = df[feature_name + '_paired'] / ( df[feature_name + '_paired'] + df[feature_name + '_unpaired_min'] * 2)
        return df.fillna(0)


class FeatureExtractor(FeatureExtractorGeneral):
    def __init__(self, net_paired, loader_paired, net_unpaired, loader_unpaired, **kwargs):
        self.net_paired = net_paired
        self.loader_paired = loader_paired
        self.net_unpaired = net_unpaired
        self.loader_unpaired = loader_unpaired
        super().__init__(**kwargs)

    def process(self, df):
        common_words_df, non_common_words_df = self.split_paired_unpaired(df, self.word_vectors, self.n_proc)
        paired, unpaired = self.generate_paired_unpaired(common_words_df, non_common_words_df)
        return self.extract_features(paired, unpaired, df)

    def generate_paired_unpaired(self, common_words_df, non_common_words_df, device='cpu'):
        X = self.loader_paired.preprocess(common_words_df, self.word_vectors)
        X.to(device)
        grouped = self.loader_paired.aggregated.copy()
        grouped['pred'] = self.net_paired(X).cpu().detach().numpy()
        tmp = common_words_df.copy()
        merge_cols = ['attribute', 'left_word', 'right_word']
        paired_words = tmp.merge(grouped[merge_cols + ['pred']], on=merge_cols, suffixes=('', ''))
        self.paired_raw = paired_words

        X = self.loader_unpaired.preprocess(non_common_words_df, self.word_vectors)
        grouped = self.loader_unpaired.aggregated.copy()
        X.to(device)
        grouped['pred'] = self.net_unpaired(X).cpu().detach().numpy()
        tmp = non_common_words_df.copy()
        merge_cols = ['attribute', 'word']
        unpaired_words = tmp.merge(grouped[merge_cols + ['pred']], on=merge_cols, suffixes=('', ''))
        self.unpaired_raw = unpaired_words
        return paired_words, unpaired_words

    @staticmethod
    def split_paired_unpaired(df, word_vectors, n_proc=2, exclude_attrs=['id', 'left_id', 'right_id', 'label']):
        cols = [x[5:] for x in df.columns if x not in exclude_attrs and x.startswith('left_')]
        tk_divider = Tokens_divider(cols, word_vectors)
        if n_proc == 1:
            tmp = [tk_divider.tokens_in_vocab_division(x) for x in df.iterrows()]
        else:
            pool = Pool(n_proc)
            tmp = pool.map(tk_divider.tokens_in_vocab_division, df.iterrows())
            pool.close()
            pool.join()
        tuples_common, tuples_non_common = [], []
        for x, y in tmp:
            tuples_common += x
            tuples_non_common += y
        common_words_df = pd.DataFrame(tuples_common)
        non_common_words_df = pd.DataFrame(tuples_non_common)
        return common_words_df, non_common_words_df

    def extract_features(self, com_df, non_com_df, df, complementary=True):
        """ old version
        stat['mean_diff'] = stat['mean_paired'] - stat['mean_unpaired']
        stat['sum_diff'] = stat['sum_paired'] - stat['sum_unpaired']
        stat['overlap'] = stat['count_paired'] / (stat['count_paired'] + stat['count_unpaired'])
        stat = stat.fillna(0)

        """
        functions = ['mean', 'sum', 'count', 'min', 'max']
        paired_stat = com_df.groupby(['id'])['pred'].agg(functions)
        paired_stat.columns += '_paired'

        tmp = non_com_df.copy()
        tmp['pred'] = ( 1 - tmp['pred']) if complementary else tmp['pred']
        stat = (tmp.groupby(['id', 'side'])['pred']).agg(functions)
        unpaired_stat = stat.unstack(1)
        unpaired_stat.columns = ['_'.join(col) for col in unpaired_stat.columns]
        unpaired_stat = unpaired_stat.fillna(0)
        unpaired_stat = self.compute_min_max_features(unpaired_stat, functions)

        stat = (tmp.groupby(['id'])['pred']).agg(functions)
        unpaired_stat_full = stat
        unpaired_stat_full = unpaired_stat_full.fillna(0)
        unpaired_stat_full.columns += '_unpaired'

        df.index.name = 'id'

        stat = paired_stat.merge(unpaired_stat, on='id', how='outer').merge(unpaired_stat_full, on='id', how='outer').merge(
            df.reset_index()[['id', 'label']], on='id', how='outer').fillna(0)
        stat = self.compute_derived_features(stat, functions)
        return stat


class FeatureExtractorOOV(FeatureExtractorGeneral):

    def process(self, non_in_vocab_df):

        com_df, non_com_df = self.generate_paired_unpaired(non_in_vocab_df)
        oov_stat = self.extract_features(com_df, non_com_df, non_in_vocab_df)
        return oov_stat

    def generate_paired_unpaired(self, df):
        if self.n_proc == 1:
            tmp_list = [ self.tokens_in_vocab_division(x) for x in df.iterrows() ]
        else:
            pool = Pool(self.n_proc)
            tmp_list = pool.map(self.tokens_in_vocab_division, df.iterrows())
            pool.close()
            pool.join()
        tmp1, tmp2 = [], []
        for x, y in tmp_list:
            tmp1 += x
            tmp2 += y
        com_df = pd.DataFrame(tmp1)
        non_com_df = pd.DataFrame(tmp2)
        com_df['is_code'] = com_df['left_word'].apply(self.is_code)
        non_com_df['is_code'] = non_com_df['word'].apply(self.is_code)
        pair_to_add = self.check_codes(non_com_df)
        com_df = pd.concat([com_df, pd.DataFrame(pair_to_add)])
        self.paired_raw = com_df
        self.unpaired_raw = non_com_df
        return com_df, non_com_df

    def tokens_in_vocab_division(self, item):
        index, el = item
        label = el['label']
        el_words = {}
        for col in self.cols:
            if el.isnull()[[self.lp + col, self.rp + col]].any():  # both description must be non Nan
                el_words[self.lp + col] = []
                el_words[self.rp + col] = []
            else:
                el_words[self.lp + col] = str(el[self.lp + col]).split()
                el_words[self.rp + col] = str(el[self.rp + col]).split()
        common, non_common = self.split_paired_unpaired_tokens(el_words)
        return self.to_list_of_dict(common, non_common, index, label)

    def split_paired_unpaired_tokens(self, el_words):
        common_words = {col: [] for col in self.cols}
        non_common_words = {col: [] for col in self.all_cols}
        for col in self.cols:
            l_words, r_words = el_words[self.lp + col], el_words[self.rp + col]
            paired = []
            for word in l_words:
                if word in r_words:
                    common_words[col] += [(word, word)]
                    paired.append(word)
                else:
                    non_common_words[self.lp + col] += [word]
            for word in np.setdiff1d(r_words, paired):
                non_common_words[self.rp + col] += [word]
        return common_words, non_common_words

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
            prefix = self.lp
            for word in non_common_words[prefix + col]:
                tmp_dict.update(word=word)
                tuples_non_common.append(tmp_dict.copy())

            tmp_dict.update(side='right')
            prefix = self.rp
            for word in non_common_words[prefix + col]:
                tmp_dict.update(word=word)
                tuples_non_common.append(tmp_dict.copy())
        return tuples_common, tuples_non_common

    def is_code(self, word):
        tmp_word = word.replace('-', '')
        if len(tmp_word) < 4:
            return False
        tmp = re.search(r'(\d+[A-Za-z]+)|([A-Za-z]+\d+)', tmp_word)
        if tmp is None:
            return False
        tmp_words = re.findall(r'(?P<letters>[A-Za-z]+)', tmp_word)
        if len(tmp_words) > 1:
            return True
        if not self.word_vectors.__contains__(tmp_words[0].lower()):
            return True
        return False

    @staticmethod
    def check_codes(df):
        """
        Find codes that are semantically the same but are not identically
        e.g. : a code is contained in another --> 'dslra200kw' vs 'dslra200k'
            or codes differs by a '-' --> 'dslra200' vs 'dslra-200'
        """
        all_attr = df
        pair_to_add = []
        for attr in all_attr.attribute.unique():
            x = all_attr[all_attr.attribute == attr]
            id_code_l = x[(x.is_code == True) & (x.side == 'left')].id.values
            id_code_r = x[(x.is_code == True) & (x.side == 'right')].id.values
            for index in np.intersect1d(id_code_l, id_code_r):
                df_id = x[(x.id == index) & (x.is_code == True)]
                l_codes = df_id[(df_id.side == 'left')].word.values
                r_codes = df_id[(df_id.side == 'right')].word.values
                l_codes_replaced = [code.replace('-', '') for code in l_codes]
                r_codes_replaced = [code.replace('-', '') for code in r_codes]
                for r_idx, r_code in enumerate(r_codes_replaced):
                    for l_idx, l_code in enumerate(l_codes_replaced):
                        if len(r_code) < len(l_code):
                            contained = r_code in l_code
                        else:
                            contained = l_code in r_code
                        if contained:
                            tmp_dict = df_id[df_id.word == l_codes[l_idx]].iloc[0].to_dict()
                            tmp_dict.pop('side')
                            tmp_dict.pop('word')
                            tmp_dict['left_word'] = l_code
                            tmp_dict['right_word'] = r_code
                            pair_to_add.append(tmp_dict.copy())
                            l_codes_replaced.remove(l_code)
                            all_attr.drop(df_id[df_id.word == l_codes[l_idx]].index, inplace=True)
                            all_attr.drop(df_id[df_id.word == r_codes[r_idx]].index, inplace=True)
                            break
        return pair_to_add

    def extract_features(self, com_df, non_com_df, df):
        paired_stat = com_df.groupby(['id']).agg(
            {'is_code': ['count', ('n_code', lambda x: x[x == True].count())]}).droplevel(0, 1)
        paired_stat.columns += '_paired'

        tmp = non_com_df.groupby(['id', 'side']).agg(
            {'is_code': ['count', ('n_code', lambda x: x[x == True].count())]}).unstack(1).droplevel(0, 1)
        tmp.columns = [x + '_' + y for x, y in tmp.columns]
        unpaired_stat = tmp
        columns_to_compute = ['n_code', 'count']
        unpaired_stat = self.compute_min_max_features(unpaired_stat, columns_to_compute)

        stat = non_com_df.groupby(['id']).agg(
            {'is_code': ['count', ('n_code', lambda x: x[x == True].count())]}).droplevel(0, 1)
        unpaired_stat_full = stat
        unpaired_stat_full = unpaired_stat_full.fillna(0)
        unpaired_stat_full.columns += '_unpaired'

        stat = unpaired_stat.merge(paired_stat, on='id', how='outer').merge(unpaired_stat_full, on='id', how='outer').fillna(0)

        stat = self.compute_derived_features(stat, columns_to_compute)
        stat.columns += '_oov'
        df.index.name = 'id'
        return stat.merge(df.reset_index()[['id','label']], on='id', how='outer').fillna(0).astype(int).rename(columns={'label_oov':'label'})
