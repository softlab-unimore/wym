import numpy as np
import pandas as pd


class FeatureExtractorGeneral:

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
    def compute_derived_features(df, feature_names, possible_unpaired=['_exclusive', '', '_both']):
        # possible_upaired = ['_max', '_min', ''] # to differentiate between lef and right unpaired elements
        for feature_name in feature_names:
            for x in possible_unpaired:
                if x in ['_max', '_min']:
                    suffix = '_min' if x == '_max' else '_max'
                else:
                    suffix = ''
                df[feature_name + '_diff' + suffix + x] = df[feature_name + '_paired'] - df[
                    feature_name + '_unpaired' + x]
                df[feature_name + '_perc' + suffix + x] = (df[feature_name + '_paired'] + 1e-9) / (
                        1e-9 + df[feature_name + '_paired'] + df[
                    feature_name + '_unpaired' + x])
            # df[feature_name + '_diff' + '_2min'] = df[feature_name + '_paired'] - (df[feature_name + '_unpaired_min'] * 2)
            # df[feature_name + '_perc' + '_2min'] = df[feature_name + '_paired'] / ( df[feature_name + '_paired'] + df[feature_name + '_unpaired_min'] * 2)
        return df.fillna(0)


class FeatureExtractor(FeatureExtractorGeneral):
    @staticmethod
    def extract_features(word_pairs_df, complementary=True, pos_threshold=.40):
        functions = ['mean', 'sum', 'count', 'min', 'max']
        all_stat = word_pairs_df.groupby(['id'])['pred'].agg(functions)
        all_stat.columns += '_all'

        pos_mask = word_pairs_df.pred >= pos_threshold
        com_df, non_com_df = word_pairs_df[pos_mask], word_pairs_df[~pos_mask]

        paired_stat = com_df.groupby(['id'])['pred'].agg(functions)
        paired_stat.columns += '_paired'

        tmp = non_com_df.copy()
        tmp['comp_pred'] = (1 - tmp['pred']) if complementary else tmp['pred']

        tmp['side'] = np.where((tmp.left_word == '[UNP]')| (tmp.right_word == '[UNP]') , 'exclusive', 'both')
        stat = (tmp.groupby(['id', 'side'])['pred']).agg(functions)
        unpaired_stat = stat.unstack(1)
        unpaired_stat.columns = ['_unpaired_'.join(col) for col in unpaired_stat.columns]
        unpaired_stat = unpaired_stat.fillna(0)
        #unpaired_stat = FeatureExtractor().compute_min_max_features(unpaired_stat, functions)

        stat = (tmp.groupby(['id'])['pred']).agg(functions)
        unpaired_stat_full = stat
        unpaired_stat_full = unpaired_stat_full.fillna(0)
        unpaired_stat_full.columns += '_unpaired'


        stat = paired_stat.merge(unpaired_stat_full, on='id', how='outer').merge(
            unpaired_stat, on='id', how='outer').merge(
            all_stat, on='id', how='outer').fillna(0).sort_index()
        # .merge(unpaired_stat, on='id', how='outer')
        stat = FeatureExtractor().compute_derived_features(stat, functions)
        return stat
