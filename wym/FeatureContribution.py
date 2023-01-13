import numpy as np
import pandas as pd
from scipy.special import softmax

from .FeatureExtractor import FeatureExtractorGeneral


def get_median_index(d):
    if d.shape[0] == 0:
        return 0
    ranks = d.rank(pct=True)
    close_to_median = abs(ranks - 0.5)
    return close_to_median.idxmin()


class FeatureContributionGeneral:
    @staticmethod
    def compute_min_max_features(df: pd.DataFrame, stat_df: pd.DataFrame,  # count_stat: pd.DataFrame,
                                 rs_col='comp_pred', features=['mean', 'sum', 'count', 'median', 'min', 'max', 'M-m'],
                                 null_value=0):
        res = []
        to_add = None
        columns = features
        if columns[0] + '_left' not in df.columns and columns[0] + '_right' not in df.columns:
            for side in ['_left', '_right']:
                for col in columns:
                    df[col + side] = null_value
        elif columns[0] + '_left' not in df.columns:
            to_add, present = '_left', '_right'
        elif columns[0] + '_right' not in df.columns:
            to_add, present = '_right', '_left'
        if to_add is not None:
            for col in columns:
                df[col + to_add] = df[col + present]
        grouped = df.groupby('id').agg('sum')
        for col in columns:
            mask = grouped[col + '_left'] < grouped[col + '_right']
            res.append(pd.Series(
                np.where(mask[df.id], df[col + '_left'], df[col + '_right']), name=col + '_unpaired_min'))
            res.append(pd.Series(
                np.where((~mask)[df.id],
                         df[col + '_left'], df[col + '_right']), name=col + '_unpaired_max'))
        res = pd.concat(res, axis=1)
        res.index = df.index
        return res

        # for greater_cond, suffix in [[True, '_max'], [False, '_min']]:
        #     df.loc[:, 'M-m' + '_unpaired' + suffix] = 0
        #     for col in features:
        #         df.loc[:, col + '_unpaired' + suffix] = 0
        #
        #         for side, opposite_side in [['left', 'right'], ['right', 'left']]:
        #             side_mask = df['side'] == side
        #             df.loc[side_mask, col + '_unpaired' + suffix] = 0
        #             ok_index = stat_df[(stat_df[col + '_' + side] > stat_df[col + '_' + opposite_side]) == greater_cond].index
        #             turn_mask = side_mask & df.id.isin(ok_index)
        #             if col == 'mean':
        #                 df.loc[turn_mask, col + '_unpaired' + suffix] = df[rs_col] / count_stat['count_' + side][
        #                     df.id].values
        #             elif col == 'sum':
        #                 df.loc[turn_mask, col + '_unpaired' + suffix] = df[rs_col]
        #             elif col == 'count':
        #                 df.loc[turn_mask, col + '_unpaired' + suffix] = 1
        #             elif col in ['max', 'min', 'median']:
        #                 tmp_mask = df.index.isin(count_stat['idx' + col + '_' + side].values)
        #                 df.loc[turn_mask & tmp_mask, col + '_unpaired' + suffix] += df.loc[tmp_mask, rs_col]
        #     if 'M-m' in features:
        #         df.loc[:, 'M-m' + '_unpaired' + suffix] = df.loc[:, 'max' + '_unpaired' + suffix] - df.loc[:, 'min' + '_unpaired' + suffix]
        # return df

    @staticmethod
    def compute_derived_features(df: pd.DataFrame, feature_names, possible_unpaired=['_exclusive', '', '_both'],
                                 null_value=0, rs_col='pred', additive_only=False):
        base_index_series = pd.Series(0, df.id.unique())

        for turn_suffix in possible_unpaired:
            for feature in feature_names:
                matched_col, unmatched_col = feature + '_paired', feature + '_unpaired' + turn_suffix
                matched_mask = df[matched_col] != 0
                unmatched_mask = df[unmatched_col] != 0
                if additive_only is False:
                    operator = '_diff'
                    new_col = feature + operator + turn_suffix
                    if feature in ['sum', 'mean', 'count', 'median', 'max', 'min', 'M-m']:
                        df.loc[matched_mask, new_col] = df.loc[matched_mask, matched_col]
                        df.loc[unmatched_mask, new_col] = -1 * df.loc[unmatched_mask, unmatched_col]
                if additive_only is False:
                    operator = '_perc'
                    new_col = feature + operator + turn_suffix
                    if feature in ['sum', 'mean', 'count', 'median', 'max', 'min', 'M-m']:
                        A_grouped = df[matched_mask].groupby('id')[matched_col].agg('sum').replace(np.nan, 0)
                        A = A_grouped.combine_first(base_index_series).to_numpy()
                        B_grouped = df[unmatched_mask].groupby('id')[unmatched_col].agg('sum').replace(np.nan, 0)
                        B = B_grouped.combine_first(base_index_series).to_numpy()
                        matched_impact = (A * A + A * B + B * B) / ((A + B) * (A + B) + 1e-9)
                        unmatched_impact = (B * B) / ((A + B) * (A + B) + 1e-9)
                        matched_impact[(unmatched_impact == 1)] = 0
                        unmatched_impact[(unmatched_impact == 1)] = 0

                        base_contributions = (df[matched_col] - df[unmatched_col])
                        pos_mask = base_contributions > 0
                        neg_mask = base_contributions < 0
                        if (pos_mask != matched_mask).all():
                            matched_sum = A
                            unmatched_sum = B
                        else:  # spread pos impact only on pos impact decision unit
                            matched_sum = base_contributions[pos_mask].groupby(df.loc[pos_mask, 'id']).agg(
                                'sum').combine_first(
                                base_index_series).to_numpy()
                            unmatched_sum = -1 * base_contributions[neg_mask].groupby(df.loc[neg_mask, 'id']).agg(
                                'sum').combine_first(
                                base_index_series).to_numpy()
                        matched_factor = pd.Series(matched_impact / (matched_sum + 1e-9),
                                                   index=base_index_series.index).fillna(
                            0)
                        unmatched_factor = pd.Series(unmatched_impact / (unmatched_sum + 1e-9),
                                                     index=base_index_series.index).fillna(0)
                        # assert unmatched_impact[58] !=0

                        df.loc[pos_mask, new_col] = base_contributions[pos_mask].values * matched_factor.loc[
                            df[pos_mask].id.values].values
                        df.loc[neg_mask, new_col] = base_contributions[neg_mask].values * unmatched_factor.loc[
                            df[neg_mask].id.values].values

        return df.fillna(0)  # df.describe().loc['max','pred':].max()


class FeatureContribution(FeatureContributionGeneral):

    @staticmethod
    def extract_features_by_attr(word_pairs_df: pd.DataFrame, attributes, complementary=True, pos_threshold=.5,
                                 null_value=0, additive_only=False):
        stat_list = []
        contribution_df = word_pairs_df[['id', 'left_word', 'right_word', 'cos_sim', 'left_attribute',
                                         'right_attribute', 'label', 'pred']]

        # OR no mixed attr but a measure for each attribute divided by left and right
        for side in ['left', 'right']:
            for attr in attributes:
                tmp_pairs = word_pairs_df.query(f'{side}_attribute == "{attr}"')
                tmp_stat = FeatureContribution.extract_features_simplified(tmp_pairs, complementary, pos_threshold,
                                                                           null_value, additive_only=additive_only)
                tmp_stat.columns = tmp_stat.columns + f'_{side}_{attr}'
                stat_list.append(tmp_stat)
                # display(tmp_stat)
        # plus overlall features
        tmp_stat = FeatureContribution.extract_features(word_pairs_df, complementary, pos_threshold, null_value,
                                                        additive_only=additive_only)
        tmp_stat.columns = tmp_stat.columns + f'_allattr'
        stat_list.append(tmp_stat)
        joined_df = word_pairs_df[['id']].join(stat_list).fillna(null_value)
        return joined_df.drop('id', 1)

    @staticmethod
    def cycle_features(df_stat_suffix_list, features, rs_col='pred', use_softmax='proportional'):
        for df, stat, pair_suffix in df_stat_suffix_list:
            turn_rs_col = rs_col if pair_suffix in ['_paired', '_all'] else 'comp_' + rs_col
            name = 'mean'
            if name in features:
                if 'count' + pair_suffix in stat.columns:
                    tmp_values = df[turn_rs_col] / stat['count' + pair_suffix][df.id].values
                else:
                    tmp_values = 0
                df[name + pair_suffix] = tmp_values
            name = 'sum'
            if name in features:
                df[name + pair_suffix] = df[turn_rs_col]
            name = 'count'
            if name in features:
                df[name + pair_suffix] = 1
            # assert df.shape[0] > 0
            df['M-m' + pair_suffix] = 0
            for name in ['max', 'min', 'median']:
                if name in features:
                    f = lambda x: softmax(x * 10)
                    if name == 'min':
                        f = lambda x: softmax(-x * 10)
                    if use_softmax is True:
                        if name != 'median':
                            weights = df.groupby('id')[turn_rs_col].transform(f)
                        elif name == 'median':
                            proportion = - np.abs(df[turn_rs_col] - values)
                            weights = proportion.groupby(df['id']).transform(f)
                        values = df.groupby('id')[turn_rs_col].transform(name)
                        df[name + pair_suffix] = values * weights
                    elif use_softmax == 'proportional':
                        values = df.groupby('id')[turn_rs_col].transform(name)
                        if name in ['min', 'max']:
                            proportion = df[turn_rs_col] / values
                        elif name == 'median':
                            proportion = - np.abs(df[turn_rs_col] - values)
                        weights = proportion.groupby(df['id']).transform(f)

                        df[name + pair_suffix] = values * weights
                    else:
                        df[name + pair_suffix] = 0
                        if 'idx' + name + pair_suffix in stat.columns:
                            tmp_mask = df.index.isin(stat['idx' + name + pair_suffix].values)
                            df.loc[tmp_mask, name + pair_suffix] = df.loc[tmp_mask, turn_rs_col]
            if 'M-m' in features:
                df['M-m' + pair_suffix] = df['max' + pair_suffix] - df['min' + pair_suffix]

    @staticmethod
    def get_contrib_functions(additive_only=False, use_softmax=True):
        if additive_only:
            functions, function_names = FeatureExtractorGeneral.functions_dict['additive']
        else:
            functions, function_names = FeatureExtractorGeneral.functions_dict['all']
        contrib_functions = []
        if 'sum' in function_names or 'mean' in function_names or 'count' in function_names:
            contrib_functions.append('count')
        if 'max' in function_names or 'M-m' in function_names:
            contrib_functions.append('idxmax')
        if 'min' in function_names or 'M-m' in function_names:
            contrib_functions.append('idxmin')
        if 'median' in function_names:
            contrib_functions.append(('idxmedian', get_median_index))
            'count', 'idxmax', 'idxmin', ('idxmedian', get_median_index)
        return contrib_functions, function_names

    @staticmethod
    def extract_features_simplified(word_pairs_df: pd.DataFrame, complementary=True, pos_threshold=.5, null_value=0,
                                    rs_col='pred', scaled=True, additive_only=False):
        # functions = ['mean', 'sum', 'min', 'max', ('M-m', lambda x: x.max() - x.min())]
        # function_names = ['mean', 'sum', 'min', 'max', 'M-m']
        contrib_functions, function_names = FeatureContribution.get_contrib_functions(additive_only)

        neg_mask = (word_pairs_df[rs_col] < pos_threshold) | (word_pairs_df.left_word == '[UNP]') | (
                word_pairs_df.right_word == '[UNP]')
        com_df, non_com_df = word_pairs_df[~neg_mask].copy(), word_pairs_df[neg_mask].copy()

        if scaled:
            com_df[rs_col] = (com_df[rs_col] - 0.5) * 2
        paired_stat = com_df.groupby(['id'])[rs_col].agg(contrib_functions)
        paired_stat.columns += '_paired'
        if scaled:
            non_com_df[rs_col] = non_com_df[rs_col] * 2
        non_com_df['comp_pred'] = (1 - non_com_df[rs_col]) if complementary else non_com_df[rs_col]
        unpaired_stat = (non_com_df.groupby(['id'])['comp_pred']).agg(contrib_functions)
        unpaired_stat.columns += '_unpaired'
        FeatureContribution.cycle_features([[com_df, paired_stat, '_paired'], [non_com_df, unpaired_stat, '_unpaired']],
                                           features=function_names)
        contribution_df = pd.concat([com_df, non_com_df], 0).sort_index().fillna(0)

        stat = FeatureContribution().compute_derived_features(contribution_df, function_names, possible_unpaired=[''],
                                                              additive_only=additive_only)
        columns = np.setdiff1d(stat.columns, list(word_pairs_df.columns) + ['comp_pred'])
        return stat.loc[:, columns].sort_index()

    @staticmethod
    def extract_features(word_pairs_df: pd.DataFrame, complementary=True, pos_threshold=.5, null_value=0,
                         rs_col='pred', scaled=True, additive_only=False):
        functions, function_names = FeatureContribution.get_contrib_functions(additive_only)
        columns_to_delete = ['left_word', 'right_word', 'cos_sim', 'left_attribute',
                             'right_attribute', 'label', 'id', 'label_corrected',
                             'label_corrected_mean', rs_col, 'comp_pred', 'side', 'scaled_p', 'token_contribution']
        columns_to_delete = np.union1d(columns_to_delete, word_pairs_df.columns)
        word_pairs_df = word_pairs_df.copy()
        all_stat = word_pairs_df.groupby(['id'])[rs_col].agg(functions)
        all_stat.columns += '_all'

        neg_mask = (word_pairs_df[rs_col] < pos_threshold) | (word_pairs_df.left_word == '[UNP]') | (
                word_pairs_df.right_word == '[UNP]')
        com_df, non_com_df = word_pairs_df[~neg_mask].copy(), word_pairs_df[neg_mask].copy()

        if scaled:
            com_df[rs_col] = (com_df[rs_col] - 0.5) * 2
        paired_stat = com_df.groupby(['id'])[rs_col].agg(functions)
        paired_stat.columns += '_paired'

        if scaled:
            non_com_df[rs_col] = non_com_df[rs_col] * 2
        tmp = non_com_df
        tmp['comp_pred'] = (1 - tmp[rs_col]) if complementary else tmp[rs_col]
        tmp['side'] = np.where((tmp.left_word == '[UNP]') | (tmp.right_word == '[UNP]'), 'exclusive', 'both')
        unpaired_exclusive = tmp[tmp['side'] == 'exclusive'].copy()
        unpaired_both = tmp[tmp['side'] == 'both'].copy()
        stat = tmp.groupby(['id', 'side'])['comp_pred'].agg(functions)
        unpaired_stat = stat.unstack(1)
        unpaired_stat.columns = ['_unpaired_'.join(col) for col in unpaired_stat.columns]

        non_com_dict = {}
        non_com_dict['both'] = tmp[tmp['side'] == 'both']
        non_com_dict['exclusive'] = tmp[tmp['side'] == 'exclusive']

        stat = (tmp.groupby(['id'])['comp_pred']).agg(functions)
        unpaired_stat_full = stat
        # unpaired_stat_full = unpaired_stat_full.fillna(0)
        unpaired_stat_full.columns += '_unpaired'

        tmp = non_com_df[(non_com_df.left_word == '[UNP]') | (non_com_df.right_word == '[UNP]')].copy()
        tmp['comp_pred'] = (1 - tmp[rs_col]) if complementary else tmp[rs_col]
        tmp['side'] = np.where((tmp.left_word == '[UNP]'), 'left', 'right')
        exclusive_df_left = tmp[tmp['side'] == 'left'].copy()
        exclusive_df_right = tmp[tmp['side'] == 'right'].copy()

        count_side_stat = (tmp.groupby(['id', 'side'])['comp_pred']).agg(functions)
        count_side_stat = count_side_stat.unstack(1)
        count_side_stat.columns = ['_'.join(col) for col in count_side_stat.columns]
        count_side_stat = count_side_stat

        FeatureContribution.cycle_features([[word_pairs_df, all_stat, '_all'],
                                            [com_df, paired_stat, '_paired'],
                                            [non_com_df, unpaired_stat_full, '_unpaired'],
                                            [unpaired_both, unpaired_stat, '_unpaired_both'],
                                            [unpaired_exclusive, unpaired_stat, '_unpaired_exclusive'],
                                            [exclusive_df_left, count_side_stat, '_left'],
                                            [exclusive_df_right, count_side_stat, '_right']
                                            ],
                                           features=function_names)

        unpaired_both = pd.concat([unpaired_both, unpaired_exclusive]).fillna(0).sort_index()
        exclusive_df = pd.concat([exclusive_df_left, exclusive_df_right]).fillna(0).sort_index()

        # functions = ['mean', 'sum', 'count', 'min', 'max', ('M-m', lambda x: x.max() - x.min()), 'median']
        # function_names = ['mean', 'sum', 'count', 'min', 'max', 'M-m', 'median']
        # tmp = word_pairs_df[(word_pairs_df.left_word == '[UNP]') | (word_pairs_df.right_word == '[UNP]')].copy()
        # tmp['comp_pred'] = (1 - tmp[rs_col]) if complementary else tmp[rs_col]
        # tmp['side'] = np.where((tmp.left_word == '[UNP]'), 'left', 'right')
        # stat = (tmp.groupby(['id', 'side'])['comp_pred']).agg(functions)
        # side_stat = stat.unstack(1)
        # side_stat.columns = ['_'.join(col) for col in side_stat.columns]
        # side_stat = side_stat.fillna(null_value).sort_index()
        exclusive_df = FeatureContributionGeneral.compute_min_max_features(df=exclusive_df, features=function_names,
                                                                           stat_df=count_side_stat,
                                                                           null_value=null_value)

        stat = word_pairs_df
        for df in [com_df, non_com_df, exclusive_df, unpaired_both]:
            columns = np.setdiff1d(df.columns, columns_to_delete)
            columns = np.setdiff1d(columns, stat.columns)
            stat = stat.join(df.loc[:, columns], how='outer').fillna(0).sort_index()
        # except Exception as e:
        #     print(e)
        #     for i, df in enumerate([paired_stat, unpaired_stat, unpaired_stat_full, all_stat, side_stat]):
        #         print(i)
        #         display(df)
        # .merge(unpaired_stat, on='id', how='outer')
        stat = FeatureContribution().compute_derived_features(stat,
                                                              function_names,
                                                              possible_unpaired=['_exclusive', '', '_both', '_min',
                                                                                 '_max'])

        columns = np.setdiff1d(stat.columns, columns_to_delete)
        return stat.loc[:, columns].sort_index()
