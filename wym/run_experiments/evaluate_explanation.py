import os
import sys

prefix = ''
if '/home/' in os.path.expanduser('~'):  # UNI env
    prefix = '/home/baraldian'
softlab_path = os.path.join(prefix + '/content/drive/Shareddrives/SoftLab/')
project_path = os.path.join(softlab_path, 'Projects', 'WYM')
sys.path.append(os.path.join(project_path, 'notebooks'))

from wym.notebook_import_utility_env import *
from warnings import simplefilter

from tqdm.autonotebook import tqdm
from Landmark_github.evaluation.Evaluate_explanation_Batch import *


class SoftlabEnv:
    sorted_dataset_names = [
        'BeerAdvo-RateBeer',
        'fodors-zagats',
        'iTunes-Amazon',
        'dirty_itunes_amazon',
        'DBLP-Scholar',
        'dirty_dblp_scholar',
        'walmart-amazon',
        'dirty_walmart_amazon',
        'DBLP-ACM',
        'dirty_dblp_acm',
        'Abt-Buy',
        'Amazon-Google',
    ]
    tasks = [
        'Structured/Beer',
        'Structured/Fodors-Zagats',
        'Structured/iTunes-Amazon',
        'Dirty/iTunes-Amazon',
        'Structured/DBLP-GoogleScholar',
        'Dirty/DBLP-GoogleScholar',
        'Structured/Walmart-Amazon',
        'Dirty/Walmart-Amazon',
        'Structured/DBLP-ACM',
        'Dirty/DBLP-ACM',
        'Textual/Abt-Buy',
        'Structured/Amazon-Google',
    ]

    def __init__(self):
        simplefilter(action='ignore', category=FutureWarning)
        simplefilter(action='ignore')
        pd.options.display.float_format = '{:.4f}'.format
        self.excluded_cols = ['id', 'left_id', 'right_id']  # ['']
        if '/home/' in os.path.expanduser('~'):  # UNI env
            prefix = os.path.expanduser('~')
        else:
            prefix = '.'
            # install here for colab env
        self.softlab_path = os.path.join(prefix + '/content/drive/Shareddrives/SoftLab/')
        self.dataset_path = os.path.join(self.softlab_path, 'Dataset', 'Entity Matching')
        self.project_path = os.path.join(self.softlab_path, 'Projects', 'WYM')
        self.base_files_path = os.path.join(self.project_path, 'dataset_files')

    @staticmethod
    def explanation_from_decision_unit_to_token(em_df, word_relevance, impact_col: str='token_contribution'):
        if impact_col not in word_relevance.columns:
            raise ValueError(f"Missing column {impact_col} in dataframe.")

        wr = append_prefix(em_df, word_relevance, decision_unit_view=True)

        df_list = list()
        wr[impact_col] = np.where((wr['left_word'] != '[UNP]') & (wr['right_word'] != '[UNP]'),
                                            wr[impact_col] / 2, wr[impact_col])
        for side in ['left', 'right']:
            side_columns = wr.columns[wr.columns.str.startswith(side)].to_list()
            token_impact = wr.loc[:, ['id', 'label', impact_col] + side_columns]
            token_impact.columns = token_impact.columns.str.replace(side + '_', '')
            token_impact = token_impact[token_impact['word'] != '[UNP]']
            token_impact['attribute'] = side + '_' + token_impact['attribute']

            df_list.append(token_impact)
        wr_flat = pd.concat(df_list, 0).reset_index(drop=True)
        wr_flat = wr_flat.rename(columns={'word_prefixes': 'word_prefix', impact_col: 'impact'})
        assert 'word_prefix' in wr_flat.columns
        return wr_flat

    @staticmethod
    def evaluate_df(word_relevance, df_to_process, predictor, exclude_attrs=('id', 'left_id', 'right_id', 'label'),
                    score_col='pred', conf_name='bert', utility='AOPC', k=5, decision_unit_view=True,
                    remove_decision_unit_only=False):
        ids = list(df_to_process.id.unique())
        print(f'Testing unit removal with -- {score_col}')
        assert df_to_process.shape[
                   0] > 0, f'DataFrame to evaluate must have some elements. Passed df has shape {df_to_process.shape[0]}'
        evaluation_df = df_to_process.copy().replace(pd.NA, '')
        word_relevance_prefix = append_prefix(evaluation_df, word_relevance, decision_unit_view=decision_unit_view,
                                              exclude_attrs=exclude_attrs)
        if score_col == 'pred':
            word_relevance_prefix['impact'] = word_relevance_prefix[score_col] - 0.5
        else:
            word_relevance_prefix['impact'] = word_relevance_prefix[score_col]
        word_relevance_prefix['conf'] = conf_name

        res_list = []
        # for side in ['left', 'right']:
        # evaluation_df['pred'] = predictor(evaluation_df)
        side_word_relevance_prefix = word_relevance_prefix.copy()
        # side_word_relevance_prefix['word_prefix'] = side_word_relevance_prefix[side + '_word_prefixes']
        # side_word_relevance_prefix = side_word_relevance_prefix.query(f'{side}_word != "[UNP]"')
        word_relevance_prefix.copy()
        ev = EvaluateExplanation(side_word_relevance_prefix, evaluation_df, predict_method=predictor,
                                 exclude_attrs=exclude_attrs, percentage=.25, num_round=3,
                                 decision_unit_view=decision_unit_view,
                                 remove_decision_unit_only=remove_decision_unit_only)

        # fixed_side = 'right' if side == 'left' else 'left'
        impacts_all = side_word_relevance_prefix
        impact_df = impacts_all[impacts_all.id.isin(ids)]
        start_el = df_to_process[df_to_process.id.isin(ids)]

        variable_side = 'all'
        res = []
        res += ev.evaluate_impacts(start_el, impact_df, variable_side, 'all', add_before_perturbation=None,
                                   add_after_perturbation=None, utility=utility, k=k)
        res_df = pd.DataFrame(res)
        res_df['conf'] = conf_name
        res_df['error'] = res_df.expected_delta - res_df.detected_delta
        res_list.append(res_df.copy())

        return pd.concat(res_list), ev

    @staticmethod
    def get_match_no_match_df(df, pred, delta=0.1, pred_threshold=0.01):
        # match_df = df[pred > .5 + delta]
        match_df = df[df.label > .5]
        sample_len = min(100, match_df.shape[0])
        match_ids = match_df.id.sample(sample_len, random_state=0).values
        # no_match_df = df[(df['label'] < 0.5) & (pred >= pred_threshold)]
        no_match_df = df[df['label'] < 0.5]
        sample_len = min(100, no_match_df.shape[0])
        no_match_ids = no_match_df.id.sample(sample_len, random_state=0).values
        return match_df, match_ids, no_match_df, no_match_ids

    def calculate_save_metric(self, comb_df, model_files_path, metric_name, prefix='', suffix='', load=False):
        prefix = prefix + '_' if prefix != '' else ''
        suffix = ('_' + suffix) if suffix != '' else ''

        try:
            os.makedirs(os.path.join(model_files_path, 'results'))
        except Exception as e:
            print(e)
        tmp_path = os.path.join(model_files_path, 'results', f'{prefix}{metric_name}_perturbations{suffix}.csv')
        if load:
            comb_df = pd.read_csv(tmp_path)
        else:
            comb_df.to_csv(tmp_path, index=False)

        if metric_name == 'AOPC':
            res_df = comb_df
            res_df['comb'] = np.where(res_df['comb_name'].str.startswith('MoRF'), 'MoRF', 'random')

            grouped = res_df.groupby(['id', 'comb']).agg(
                {'detected_delta': [('sum', lambda x: x.sum()), 'size']}).droplevel(0, 1)

            AOPC = (grouped['sum'] / grouped['size']).groupby('comb').mean()

            res_df.to_csv(tmp_path, index=False)
            tmp_path = os.path.join(model_files_path, 'results', f'{prefix}{metric_name}_score{suffix}.csv')
            AOPC.to_csv(tmp_path)
            print(AOPC)
        elif metric_name == 'sufficiency':
            res_df = comb_df
            self.res_df = res_df
            res_df['comb'] = np.where(res_df['comb_name'].str.startswith('top'), 'top', 'random')
            res_df['same_class'] = (res_df['start_pred'] > 0.5) == (res_df['new_pred'] > .5)
            sufficiency = res_df.groupby(['id', 'comb']).agg(
                {'same_class': [('sum', lambda x: x.sum() / x.size)]}).groupby('comb').mean()
            # assert False
            tmp_path = os.path.join(model_files_path, 'results', f'{prefix}{metric_name}_score{suffix}.csv')
            sufficiency.to_csv(tmp_path)
            print(sufficiency)


class WYMevaluation(SoftlabEnv):
    def __init__(self, additive_only=False):
        super().__init__()
        self.project_path = os.path.join(self.softlab_path, 'Projects', 'WYM')
        self.additive_only = additive_only
        sys.path.append(os.path.join(self.softlab_path, 'Projects/external_github/ditto'))
        sys.path.append(os.path.join(self.softlab_path, 'Projects/external_github'))
        sys.path.append(os.path.join(self.project_path, 'common_functions'))
        sys.path.append(os.path.join(self.project_path, 'src'))
        sys.path.append(os.path.join(self.project_path, 'src', 'wym'))

        # from wrapper.DITTOWrapper import DITTOWrapper
        # from landmark import Landmark

    def evaluate_all_df(self, utility='AOPC', explanation='', **kwargs):
        for df_name in tqdm(list(self.sorted_dataset_names)):
            gc.collect()
            torch.cuda.empty_cache()
            self.evaluate_single_df(df_name, utility=utility, explanation=explanation, **kwargs)

    def evaluate_single_df(self, dataset_name="Amazon-Google", reset_files=False,
                           reset_networks=False,  # @param {type:"boolean"},
                           we_finetuned=True,  # @param {type:"boolean"},
                           sentence_embedding=False, utility='AOPC', delta=.1, pred_threshold=0.01, k=5, batch_size=256,
                           explanation=''):
        model_name = 'BERT'
        model_files_path = os.path.join(self.project_path, 'dataset_files', dataset_name, model_name)
        routine, predictor = self.init_routine(dataset_name, reset_files, reset_networks, we_finetuned,
                                               sentence_embedding, chunk_size=batch_size)
        self.routine = routine
        self.predictor = predictor
        test_df = routine.test_merged.copy().replace(pd.NA, '')
        if explanation == 'LIME':
            pos_exp, neg_exp = self.load_explanations(turn_dataset_name=dataset_name, conf=explanation)
            match_ids = pos_exp.id.unique()
            no_match_ids = neg_exp.id.unique()
            decision_unit_view = False
            res_df, ev = self.evaluate_df(pos_exp,
                                          test_df[test_df.id.isin(match_ids)],
                                          predictor, score_col='impact', k=k, decision_unit_view=decision_unit_view,
                                          utility=utility)
            self.calculate_save_metric(res_df, model_files_path, metric_name=utility, suffix=explanation)

            res_df, ev = self.evaluate_df(neg_exp,
                                          test_df[test_df.id.isin(no_match_ids)],
                                          predictor, score_col='impact', k=k, decision_unit_view=decision_unit_view,
                                          utility=utility)
            self.ev = ev
            self.calculate_save_metric(res_df, model_files_path, metric_name=utility, prefix='no_match',
                                       suffix=explanation)
        else:
            decision_unit_view = True
            remove_decision_unit_only = False
            df_name = 'test'
            pred, features, word_relevance = routine.get_calculated_data(df_name)
            df = routine.test_merged.copy().replace(pd.NA, '')
            if 'decision_unit_flat' in explanation:
                word_relevance = self.explanation_from_decision_unit_to_token(df, word_relevance)
                decision_unit_view = False
            elif 'remove_du_only' in explanation:
                remove_decision_unit_only = True
                predictor = partial(routine.get_predictor(remove_decision_unit_only=True), return_data=False, lr=True,
                                    chunk_size=batch_size, reload=True)

            match_df, match_ids, no_match_df, no_match_ids = self.get_match_no_match_df(df, pred, delta=delta,
                                                                                        pred_threshold=pred_threshold)

            # predictor = lambda x : [0.5]*x.shape[0]
            print('Before')

            res_df, ev = self.evaluate_df(word_relevance[word_relevance.id.isin(match_ids)],
                                          match_df[match_df.id.isin(match_ids)],
                                          predictor, score_col='token_contribution', k=k, utility=utility,
                                          decision_unit_view=decision_unit_view,
                                          remove_decision_unit_only=remove_decision_unit_only
                                          )
            self.ev = ev
            self.calculate_save_metric(res_df, model_files_path, metric_name=utility, suffix=explanation)
            # assert False
            if utility == 'AOPC':
                return

            res_df, ev = self.evaluate_df(word_relevance[word_relevance.id.isin(no_match_ids)],
                                          no_match_df[no_match_df.id.isin(no_match_ids)],
                                          predictor, score_col='token_contribution', k=k, utility=utility,
                                          decision_unit_view=decision_unit_view,
                                          remove_decision_unit_only=remove_decision_unit_only
                                          )
            self.ev = ev
            self.calculate_save_metric(res_df, model_files_path, metric_name=utility, prefix='no_match',
                                       suffix=explanation)

    def load_explanations(self, turn_dataset_name, conf='LIME'):
        base_files_path = os.path.join(self.project_path, 'dataset_files')
        turn_files_path = os.path.join(base_files_path, turn_dataset_name, 'wym')

        tmp_path = os.path.join(turn_files_path, f'negative_explanations_{conf}.csv')
        neg_exp = pd.read_csv(tmp_path)
        tmp_path = os.path.join(turn_files_path, f'positive_explanations_{conf}.csv')
        pos_exp = pd.read_csv(tmp_path)
        print('Loaded explanations')
        return pos_exp, neg_exp

    def init_routine(self, dataset_name="Amazon-Google", reset_files=False,
                     reset_networks=False,  # @param {type:"boolean"},
                     we_finetuned=True,  # @param {type:"boolean"},
                     sentence_embedding=False, chunk_size=128, model_name="BERT", **kwargs):
        # @param ["wym"]
        softlab_path = self.softlab_path
        dataset_path = os.path.join(self.softlab_path, 'Dataset', 'Entity Matching', dataset_name)

        from wym.BERTRoutine import Routine
        finetuned_path = os.path.join(self.project_path, 'dataset_files', dataset_name, model_name, 'sBERT')

        routine = Routine(dataset_name, dataset_path, self.project_path, reset_files=reset_files,
                          reset_networks=reset_networks, softlab_path=softlab_path, model_name=model_name,
                          we_finetuned=we_finetuned, we_finetune_path=finetuned_path,
                          sentence_embedding=sentence_embedding, device='cuda', **kwargs
                          )
        routine.additive_only = self.additive_only
        routine.generate_df_embedding(chunk_size=chunk_size)
        _ = routine.compute_word_pair()

        _ = routine.net_train(
            # batch_size=512, lr=5e-5
        )
        _ = routine.preprocess_word_pairs()
        _ = routine.EM_modelling(routine.features_dict, routine.words_pairs_dict, routine.train_merged,
                                 routine.test_merged,
                                 do_feature_selection=False)

        predictor = partial(routine.get_predictor(), return_data=False, lr=True, chunk_size=chunk_size, reload=True)
        return routine, predictor

    def generate_explanation_LIME(self, **kwargs):
        for df_name in tqdm(list(self.sorted_dataset_names)):
            self.generate_explanation_single_df(df_name, **kwargs)

    def generate_explanation_single_df(self, dataset_name="Amazon-Google", reset_files=False,
                                       reset_networks=False,  # @param {type:"boolean"},
                                       we_finetuned=True,  # @param {type:"boolean"},
                                       sentence_embedding=False, num_explanations=100,
                                       num_samples=2048, batch_size=256
                                       ):
        model_name = 'BERT'
        model_files_path = os.path.join(self.project_path, 'dataset_files', dataset_name, model_name)
        routine, predictor = self.init_routine(dataset_name, reset_files, reset_networks, we_finetuned,
                                               sentence_embedding, verbose=True, chunk_size=batch_size)
        test_df = routine.test_merged.copy().replace(pd.NA, '')
        explainer = Landmark(predictor, test_df,
                             exclude_attrs=self.excluded_cols + ['label', 'id'], lprefix='left_', rprefix='right_',
                             split_expression=r' ')
        turn_df = test_df
        pos_mask = turn_df['label'] == 1
        pos_df = turn_df[pos_mask]
        neg_df = turn_df[~pos_mask]

        pos_sample = pos_df.sample(num_explanations, random_state=0) if pos_df.shape[0] >= num_explanations else pos_df
        neg_sample = neg_df.sample(num_explanations, random_state=0) if neg_df.shape[0] >= num_explanations else neg_df

        for conf in ['LIME']:  # ['single', 'double']:
            for sample, prefix in zip([pos_sample, neg_sample], ['positive', 'negative']):
                tmp_path = os.path.join(model_files_path, f'{prefix}_explanations_{conf}.csv')
                print(f'{prefix} explanations')
                try:
                    # assert False
                    tmp_df = pd.read_csv(tmp_path)
                    assert tmp_df.id.nunique() >= sample.shape[0], 'Not computed'
                    print('loaded')
                except Exception as e:
                    print(e)
                    tmp_df = explainer.explain(sample, num_samples=num_samples, conf=conf)
                    tmp_df.to_csv(tmp_path, index=False)


class DITTOevaluation(SoftlabEnv):

    def __init__(self):
        super().__init__()
        os.chdir(os.path.join(self.softlab_path, 'Projects/external_github/ditto'))

        self.checkpoint_path = os.path.join(
            os.path.join(self.softlab_path, 'Projects/external_github/ditto/checkpoints'))  # 'checkpoints'
        github_code_path = os.path.join(self.softlab_path, 'Projects/Landmark Explanation EM/Landmark_github')
        code_path = os.path.join(self.softlab_path, 'Projects/Landmark Explanation EM/Landmark code')

        sys.path.append(os.path.join(self.softlab_path, 'Projects/external_github/ditto'))
        sys.path.append(os.path.join(self.softlab_path, 'Projects/external_github'))
        sys.path.append(code_path)
        sys.path.append(github_code_path)

    def evaluate_all_df(self, utility='AOPC'):
        for df_name, task_name in tqdm(list(zip(self.sorted_dataset_names, self.tasks))):
            self.evaluate_single_df(df_name, task_name, utility=utility)

    def evaluate_single_df(self, dataset_name="Amazon-Google", task_name=None, utility='AOPC'):
        model_name = "DITTO"  # @param ["wym"]
        model_files_path = os.path.join(self.project_path, 'dataset_files', dataset_name, model_name)
        batch_size = 2048
        num_explanations = 100  # 100
        num_samples = 2048  # 2048
        task = task_name
        turn_dataset_name = dataset_name
        print('v' * 100)
        print(f'\n\n\n{task: >50}\n' + f'{turn_dataset_name: >50}\n\n\n')
        print('^' * 100)
        turn_dataset_path = os.path.join(self.dataset_path, turn_dataset_name)
        turn_files_path = os.path.join(self.base_files_path, turn_dataset_name)
        try:
            os.mkdir(turn_files_path)
        except:
            pass

        dataset_dict = {name: pd.read_csv(os.path.join(turn_dataset_path, f'{name}_merged.csv')) for name in
                        ['train', 'valid', 'test']}

        pos_exp, neg_exp = self.load_explanations(dataset_name)
        match_ids = pos_exp.id.unique()
        no_match_ids = neg_exp.id.unique()

        from wrapper.DITTOWrapper import DITTOWrapper
        model = DITTOWrapper(task, self.checkpoint_path)
        test_df = dataset_dict['test']

        # pred = model.predict(test_df)
        # delta = 0.15
        # match_df = test_df[pred > .5 + delta]
        # sample_len = min(100, match_df.shape[0])
        # match_ids = match_df.id.sample(sample_len, random_state=0).values

        # predictor = lambda x : [0.5]*x.shape[0]
        k = 5
        res_df, ev = self.evaluate_df(pos_exp,
                                      test_df[test_df.id.isin(match_ids)],
                                      model.predict, score_col='impact', k=k, decision_unit_view=False, utility=utility)
        self.calculate_save_metric(res_df, model_files_path, metric_name=utility)

        res_df, ev = self.evaluate_df(neg_exp,
                                      test_df[test_df.id.isin(no_match_ids)],
                                      model.predict, score_col='impact', k=k, decision_unit_view=False, utility=utility)
        self.calculate_save_metric(res_df, model_files_path, metric_name=utility, prefix='no_match')

    def load_explanations(self, turn_dataset_name, conf='LIME'):
        base_files_path = os.path.join(self.softlab_path, 'Projects/Landmark Explanation EM/dataset_files')
        turn_files_path = os.path.join(base_files_path, turn_dataset_name)

        tmp_path = os.path.join(turn_files_path, f'negative_explanations_{conf}.csv')
        neg_exp = pd.read_csv(tmp_path)
        tmp_path = os.path.join(turn_files_path, f'positive_explanations_{conf}.csv')
        pos_exp = pd.read_csv(tmp_path)
        print('Loaded explanations')
        return pos_exp, neg_exp


if __name__ == "__main__":
    ev = WYMevaluation()
    ev.evaluate_single_df('iTunes-Amazon', utility=True, explanation='to_delete', reset_files=False)

    exit(0)

    # Look in '''Evaluation explanation + General training wym''' for more
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--task", type=str, default="Structured/Beer")

    ev = WYMevaluation()
    ev.evaluate_all_df(utility='AOPC', explanation='last')

    ev = WYMevaluation()
    ev.evaluate_all_df(utility='sufficiency', explanation='last')

    ev = WYMevaluation()
    # ev.evaluate_all_df(utility='degradation', reset_files=True, reset_networks=True, explanation='improved')
    # ev.evaluate_all_df(utility='degradation', explanation='decision_unit_flat', batch_size=512)
    ev.evaluate_all_df(utility='degradation', explanation='last', batch_size=512)

    # ev = WYMevaluation()
    # ev.evaluate_all_df(explanation='decision_unit_flat_last')

    # ev = WYMevaluation()
    # ev.evaluate_all_df(utility='degradation', explanation='remove_du_only')

    # ev = WYMevaluation(additive_only=True)
    # ev.additive_only = True
    # ev.evaluate_all_df(utility='degradation', explanation='additive_only')

    # ev = WYMevaluation()
    # ev.generate_explanation_LIME() # LIME
    #
    # ev = WYMevaluation()
    # ev.evaluate_all_df(utility='sufficiency', explanation='LIME')
    # ev = WYMevaluation()
    # ev.evaluate_all_df(utility='degradation', explanation='LIME')

    # DITTOevaluation().evaluate_all_df(utility='sufficiency')
    # DITTOevaluation().evaluate_all_df(utility='AOPC')
    # DITTOevaluation().evaluate_all_df(utility='degradation')

    # print('-' * 10 + '>' * 5 + 'decision_unit_flat_proportional' + '<' * 5 + '-' * 10)
    # ev = WYMevaluation()
    # ev.evaluate_all_df(utility='degradation', explanation='decision_unit_flat_proportional', batch_size=512)
    #
    # print('-' * 10 + '>' * 5 + 'proportional' + '<' * 5 + '-' * 10)
    # ev = WYMevaluation()
    # ev.evaluate_all_df(utility='degradation', explanation='proportional', batch_size=512)
