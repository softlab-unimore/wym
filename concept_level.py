import pandas as pd
import torch
from multiprocessing import Pool
from .Token_divider import Tokens_divider


class PreProcess():

    def __init__(self, net_paired, loader_paired, net_unpaired, loader_unpaired, df, exclude_attrs=['id', 'left_id', 'right_id','label']):
        self.net_paired = net_paired
        self.loader_paired = loader_paired
        self.net_unpaired = net_unpaired
        self.loader_unpaired = loader_unpaired
        self.cols = [ x[5:] for x in df.columns if x not in exclude_attrs and x.startswith('left_')]

    def split_paired_unpaired(self, df):
        all_cols = ['left_' + col for col in self.cols] + ['right_' + col for col in self.cols]
        tk_divider = tk_divider = Tokens_divider(self.cols, all_cols)
        pool = Pool(4)
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


    def preprocess(self, df):
        common_words_df, non_common_words_df =self.split_paired_unpaired(df)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        X = self.loader_paired.preprocess(common_words_df)
        grouped = self.loader_paired.aggregated[:]
        grouped['pred'] = self.net_paired(X.to(device)).cpu().detach().numpy()
        tmp = common_words_df[:]
        merge_cols = ['attribute', 'left_word', 'right_word']
        paired_words = tmp.merge(grouped[merge_cols + ['pred']], on=merge_cols, suffixes=('', ''))

        X = self.loader_unpaired.preprocess(non_common_words_df)
        grouped = self.loader_unpaired.aggregated[:]
        grouped['pred'] = self.net_unpaired(X.to(device)).cpu().detach().numpy()
        tmp = non_common_words_df[:]
        merge_cols = ['attribute', 'word']
        unpaired_words = tmp.merge(grouped[merge_cols + ['pred']], on=merge_cols, suffixes=('', ''))

        functions = ['mean', 'sum', 'count']
        paired_stat = paired_words.groupby(['id'])['pred'].agg(functions)
        tmp = unpaired_words[:]
        tmp['pred'] = 1 - tmp['pred']
        stat = (tmp.groupby(['id', 'side'])['pred']).agg(functions)
        unpaired_stat = stat.unstack(1)
        unpaired_stat.columns = ['_'.join(col) for col in unpaired_stat.columns]
        unpaired_stat = unpaired_stat.fillna(0)
        left_minor = unpaired_stat['count_left'] < unpaired_stat['count_right']
        tmp = unpaired_stat[left_minor][pd.Index(functions) + '_left']
        tmp.columns = functions
        tmp2 = unpaired_stat[~left_minor][pd.Index(functions) + '_right']
        tmp2.columns = functions
        unpaired_stat = pd.concat([tmp, tmp2])
        paired_stat.columns += '_paired'
        unpaired_stat.columns += '_unpaired'

        stat = paired_stat.merge(unpaired_stat, on='id', how='outer').merge(df[['id', 'label']], on='id',
                                                                            how='outer').fillna(0)
        stat['mean_diff'] = stat['mean_paired'] - stat['mean_unpaired']
        stat['sum_diff'] = stat['sum_paired'] - stat['sum_unpaired']
        stat['overlap'] = stat['count_paired'] / (stat['count_paired'] + stat['count_unpaired'])
        stat = stat.fillna(0)
        return stat

