import numpy as np
import pandas as pd


class FeatureExtractorGeneral:
    functions_dict = {'additive': [['mean', 'sum', 'count'], ['mean', 'sum', 'count']],
                      'all':[ ['mean', 'sum', 'count', 'min', 'max', ('M-m', lambda x: x.max() - x.min()), 'median'],
                              ['mean', 'sum', 'count', 'min', 'max', 'M-m', 'median']]}


    @staticmethod
    def compute_min_max_features(df: pd.DataFrame, columns, null_value=0):
        res = []
        to_add = None
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
        for col in columns:
            res.append(pd.Series(
                np.where(df[col + '_left'] < df[col + '_right'],
                         df[col + '_left'], df[col + '_right']), name=col + '_unpaired_min'))
            res.append(pd.Series(
                np.where(df[col + '_left'] > df[col + '_right'],
                         df[col + '_left'], df[col + '_right']), name=col + '_unpaired_max'))
        res = pd.concat(res, axis=1)
        res.index = df.index
        return res

    @staticmethod
    def compute_derived_features(df: pd.DataFrame, feature_names, possible_unpaired=['_exclusive', '', '_both'],
                                 null_value=0, additive_only=False):
        # possible_upaired = ['_max', '_min', ''] # to differentiate between lef and right unpaired elements
        for feature_name in feature_names:
            for x in possible_unpaired:
                if additive_only is False:
                    df[feature_name + '_perc' + x] = (df[feature_name + '_paired']) / (
                        1e-9 + df[feature_name + '_paired'] + df[
                        feature_name + '_unpaired' + x])

                    df[feature_name + '_diff' + x] = df[feature_name + '_paired'] - df[
                        feature_name + '_unpaired' + x]

            # df[feature_name + '_diff' + '_2min'] = df[feature_name + '_paired'] - (df[feature_name + '_unpaired_min'] * 2)
            # df[feature_name + '_perc' + '_2min'] = df[feature_name + '_paired'] / ( df[feature_name + '_paired'] + df[feature_name + '_unpaired_min'] * 2)
        return df.fillna(null_value)


class FeatureExtractor(FeatureExtractorGeneral):

    @staticmethod
    def extract_features_by_attr(word_pairs_df: pd.DataFrame, attributes, complementary=True, pos_threshold=.5,
                                 null_value=0, additive_only=False):
        stat_list = []
        # for attr in attributes:
        #     tmp_pairs = word_pairs_df.query(f'left_attribute == "{attr}" & right_attribute == "{attr}"')
        #     tmp_stat = FeatureExtractor.extract_features(tmp_pairs, complementary, pos_threshold, null_value)
        #     stat.append(tmp_stat)
        # tmp_pairs = word_pairs_df.query(f'left_attribute != right_attribute')
        # tmp_stat = FeatureExtractor.extract_features(tmp_pairs, complementary, pos_threshold, null_value)
        # stat.append(tmp_stat)

        # OR no mixed attr but a measure for each attribute divided by left and right
        for side in ['left', 'right']:
            for attr in attributes:
                tmp_pairs = word_pairs_df.query(f'{side}_attribute == "{attr}"')
                tmp_stat = FeatureExtractor.extract_features_simplified(tmp_pairs, complementary, pos_threshold,
                                                                        null_value, additive_only=additive_only)
                tmp_stat.columns = tmp_stat.columns + f'_{side}_{attr}'
                stat_list.append(tmp_stat)
        # plus overlall features
        tmp_stat = FeatureExtractor.extract_features(word_pairs_df, complementary, pos_threshold, null_value,
                                                     additive_only=additive_only)
        tmp_stat.columns = tmp_stat.columns + f'_allattr'
        stat_list.append(tmp_stat)
        # side_features = pd.concat(stat_list, 1).fillna(null_value)
        # features = [x.replace('left_', '') for x in side_features.columns if x.startswith('left_')]
        # tmp_stat = pd.DataFrame(index=side_features.index)
        # for x in features:
        #     tmp_stat[x] = (side_features['left_' + x] + side_features['right_' + x]) / 2
        # all_stat = FeatureExtractor.extract_features(word_pairs_df, complementary, pos_threshold, null_value)
        # all_stat.columns = all_stat.columns + f'_allattr'
        # return pd.concat([tmp_stat, all_stat], 1).fillna(null_value)

        return pd.concat(stat_list, 1).fillna(null_value).sort_index()

    @staticmethod
    def extract_features_simplified(word_pairs_df: pd.DataFrame, complementary=True, pos_threshold=.5, null_value=0,
                                    scaled=True, additive_only=False):
        if additive_only:
            functions, function_names = FeatureExtractorGeneral.functions_dict['additive']
        else:
            functions, function_names = FeatureExtractorGeneral.functions_dict['all']
        word_pairs_df = word_pairs_df.copy()

        neg_mask = (word_pairs_df.pred < pos_threshold) | (word_pairs_df.left_word == '[UNP]') | (
                word_pairs_df.right_word == '[UNP]')
        com_df, non_com_df = word_pairs_df.loc[~neg_mask, :].copy(), word_pairs_df.loc[neg_mask, :].copy()

        if scaled:
            com_df['pred'] = (com_df['pred'] - 0.5) * 2
        paired_stat = com_df.groupby(['id'])['pred'].agg(functions)
        paired_stat.columns += '_paired'

        if scaled:
            non_com_df['pred'] = non_com_df['pred'] * 2
        tmp = non_com_df
        tmp['comp_pred'] = (1 - tmp['pred']) if complementary else tmp['pred']

        stat = (tmp.groupby(['id'])['comp_pred']).agg(functions)
        unpaired_stat_full = stat
        unpaired_stat_full = unpaired_stat_full.fillna(null_value)
        unpaired_stat_full.columns += '_unpaired'

        # try:
        stat = paired_stat
        for df in [unpaired_stat_full]:
            stat = stat.merge(df, on='id', how='outer').sort_index()
            if 'id' in stat.columns:
                stat = stat.set_index('id')

        # except Exception as e:
        #     for i, df in enumerate([paired_stat, unpaired_stat, unpaired_stat_full, all_stat, side_stat]):
        #         print(i)
        #         display(df)
        # .merge(unpaired_stat, on='id', how='outer')
        stat = FeatureExtractor().compute_derived_features(stat.fillna(null_value), function_names,
                                                           possible_unpaired=[''], additive_only=additive_only)
        if 'id' in stat.columns:
            stat = stat.set_index('id')
        stat = stat.sort_index()
        return stat


    @staticmethod
    def extract_features(word_pairs_df: pd.DataFrame, complementary=True, pos_threshold=.5, null_value=0, scaled=True,
                         additive_only=False):
        if additive_only:
            functions, function_names = FeatureExtractorGeneral.functions_dict['additive']
        else:
            functions, function_names = FeatureExtractorGeneral.functions_dict['all']
        word_pairs_df = word_pairs_df.copy()
        all_stat = word_pairs_df.groupby(['id'])['pred'].agg(functions)
        all_stat.columns += '_all'

        neg_mask = (word_pairs_df.pred < pos_threshold) | (word_pairs_df.left_word == '[UNP]') | (
                word_pairs_df.right_word == '[UNP]')
        com_df, non_com_df = word_pairs_df[~neg_mask].copy(), word_pairs_df[neg_mask].copy()

        if scaled:
            com_df['pred'] = (com_df['pred'] - 0.5) * 2
        paired_stat = com_df.groupby(['id'])['pred'].agg(functions)
        paired_stat.columns += '_paired'

        if scaled:
            non_com_df['pred'] = non_com_df['pred'] * 2
        non_com_df['comp_pred'] = (1 - non_com_df['pred']) if complementary else non_com_df['pred']
        non_com_df['side'] = np.where((non_com_df.left_word == '[UNP]') | (non_com_df.right_word == '[UNP]'),
                                      'exclusive', 'both')
        stat = non_com_df.groupby(['id', 'side'])['comp_pred'].agg(functions)
        unpaired_stat = stat.unstack(1)
        unpaired_stat.columns = ['_unpaired_'.join(col) for col in unpaired_stat.columns]
        if 'mean_unpaired_both' not in unpaired_stat.columns:
            for col in function_names:
                unpaired_stat[col + '_unpaired_both'] = null_value
        if 'mean_unpaired_exclusive' not in unpaired_stat.columns:
            for col in function_names:
                unpaired_stat[col + '_unpaired_exclusive'] = null_value
        unpaired_stat = unpaired_stat.fillna(null_value)
        # unpaired_stat.index.name='id'

        stat = (non_com_df.groupby(['id'])['comp_pred']).agg(functions)
        unpaired_stat_full = stat
        unpaired_stat_full = unpaired_stat_full.fillna(null_value)
        unpaired_stat_full.columns += '_unpaired'

        non_com_df = non_com_df[(non_com_df.left_word == '[UNP]') | (non_com_df.right_word == '[UNP]')].copy()
        non_com_df['comp_pred'] = (1 - non_com_df['pred']) if complementary else non_com_df['pred']
        non_com_df['side'] = np.where((non_com_df.left_word == '[UNP]'), 'left', 'right')
        stat = (non_com_df.groupby(['id', 'side'])['comp_pred']).agg(functions)
        side_stat = stat.unstack(1)
        side_stat.columns = ['_'.join(col) for col in side_stat.columns]
        side_stat = side_stat.fillna(null_value)
        side_stat = FeatureExtractorGeneral.compute_min_max_features(side_stat, function_names, null_value=null_value)
        side_stat.index.name = 'id'

        # try:
        stat = paired_stat
        for df in [unpaired_stat_full, unpaired_stat, all_stat, side_stat]:
            df.index.name = 'id'
            stat = stat.merge(df, on='id', how='outer').sort_index()
            if 'id' in stat.columns:
                stat = stat.set_index('id')
        # except Exception as e:
        #     print(e)
        #     for i, df in enumerate([paired_stat, unpaired_stat, unpaired_stat_full, all_stat, side_stat]):
        #         print(i)
        #         display(df)
        # .merge(unpaired_stat, on='id', how='outer')
        stat = FeatureExtractor().compute_derived_features(stat,
                                                           function_names,
                                                           possible_unpaired=['_exclusive', '', '_both', '_min',
                                                                              '_max'], additive_only=additive_only)

        if 'id' in stat.columns:
            stat = stat.set_index('id')
        stat = stat.sort_index()
        return stat

    @staticmethod
    def extract_features_min(word_pairs_df: pd.DataFrame, complementary=True, pos_threshold=.5, null_value=0,
                             scaled=True):
        print('features_min')
        functions = ['mean', 'count', ]
        function_names = ['mean', 'count']
        word_pairs_df = word_pairs_df.copy()
        all_stat = word_pairs_df.groupby(['id'])['pred'].agg(functions)
        all_stat.columns += '_all'

        neg_mask = (word_pairs_df.pred < pos_threshold) | (word_pairs_df.left_word == '[UNP]') | (
                word_pairs_df.right_word == '[UNP]')
        com_df, non_com_df = word_pairs_df[~neg_mask].copy(), word_pairs_df[neg_mask].copy()

        if scaled:
            com_df['pred'] = (com_df['pred'] - 0.5) * 2
        paired_stat = com_df.groupby(['id'])['pred'].agg(functions)
        paired_stat.columns += '_paired'

        if scaled:
            non_com_df['pred'] = non_com_df['pred'] * 2
        non_com_df['comp_pred'] = (1 - non_com_df['pred']) if complementary else non_com_df['pred']

        stat = (non_com_df.groupby(['id'])['comp_pred']).agg(functions)
        unpaired_stat_full = stat
        unpaired_stat_full = unpaired_stat_full.fillna(null_value)
        unpaired_stat_full.columns += '_unpaired'

        # try:
        stat = paired_stat
        for df in [unpaired_stat_full, all_stat]:
            df.index.name = 'id'
            stat = stat.merge(df, on='id', how='outer').sort_index()
            if 'id' in stat.columns:
                stat = stat.set_index('id')
        # except Exception as e:
        #     print(e)
        #     for i, df in enumerate([paired_stat, unpaired_stat, unpaired_stat_full, all_stat, side_stat]):
        #         print(i)
        #         display(df)
        # .merge(unpaired_stat, on='id', how='outer')

        if 'id' in stat.columns:
            stat = stat.set_index('id')
        stat = stat.fillna(null_value).sort_index()
        return stat
