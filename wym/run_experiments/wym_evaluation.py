from typing import Union
import pandas

from wym.import_utility import *
from wym.import_utility import prefix
from warnings import simplefilter
from tqdm.autonotebook import tqdm
from Landmark_github.evaluation.Evaluate_explanation_Batch import *
from wym.BERTRoutine import Routine
from Landmark_github.landmark.landmark import Mapper

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

        self.softlab_path = os.path.join(prefix + '/content/drive/Shareddrives/SoftLab/')
        self.project_path = os.path.join(self.softlab_path, 'Projects', 'WYM')
        self.base_files_path = os.path.join(self.project_path, 'dataset_files')

        self.res_df = None

    def calculate_save_metric(self, comb_df, model_files_path, metric_name, prefix='', suffix='', load=False):
        prefix = prefix + '_' if prefix != '' else ''
        suffix = ('_' + suffix) if suffix != '' else ''

        try:
            os.makedirs(os.path.join(model_files_path, 'results'))
        except FileExistsError:
            pass

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

            aopc = (grouped['sum'] / grouped['size']).groupby('comb').mean()

            res_df.to_csv(tmp_path, index=False)
            tmp_path = os.path.join(model_files_path, 'results', f'{prefix}{metric_name}_score{suffix}.csv')
            aopc.to_csv(tmp_path)
            print(aopc)

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


class WYMEvaluation(SoftlabEnv):
    def __init__(self, dataset_name: str='', dataset_df: pandas.DataFrame=None, model_name: str='BERT',
                 reset_files: bool=False, reset_networks: bool=False, we_finetuned: bool=True,
                 sentence_embedding: bool=False, batch_size: int=256, delta: float=0.1, additive_only: bool=False,
                 evaluate_removing_du: bool=True, recompute_embeddings: bool=True, variable_side: str='all',
                 fixed_side: str='all', evaluate_positive: bool=True):

        if not dataset_name and not dataset_df:
            raise ValueError("Dataset name cannot be empty if the dataset DataFrame is None.")

        if dataset_name and dataset_df:
            raise ValueError("Specifying a dataset name is incorrect if a DataFrame is passed.")

        super().__init__()
        # definition of WYM parameters
        self.project_path = os.path.join(self.softlab_path, 'Projects', 'WYM')
        self.evaluate_removing_du = evaluate_removing_du
        self.recompute_embeddings = recompute_embeddings
        self.dataset_name = dataset_name
        self.evaluate_positive = evaluate_positive
        self.delta = delta

        # definition of evaluation parameters for the explanation
        self.variable_side = variable_side
        self.fixed_side = fixed_side

        current_path_env = set(sys.path)  # speed up search for duplicates

        self.paths = {
            'ditto': os.path.join(self.softlab_path, 'Projects/external_github/ditto'),
            'external_github': os.path.join(self.softlab_path, 'Projects/external_github'),
            'common_functions': os.path.join(self.project_path, 'common_functions'),
            'wym_src': os.path.join(self.project_path, 'src'),
            'wym': os.path.join(self.project_path, 'src', 'wym')
        }

        for path_ in self.paths.values():
            if path_ not in current_path_env:
                sys.path.append(path_)

        if not dataset_df:
            # definition of environment paths
            self.dataset_path = os.path.join(self.softlab_path, 'Dataset', 'Entity Matching', dataset_name)

            # definition of BERT Routine parameters
            self.reset_files = reset_files
            self.reset_networks = reset_networks
            self.we_finetuned = we_finetuned
            self.sentence_embedding = sentence_embedding
            self.batch_size = batch_size
            self.additive_only = additive_only

            # definition of path for the dataset
            self.model_name = model_name
            self.model_files_path = os.path.join(self.project_path, 'dataset_files', dataset_name, self.model_name)

            # definition of BERT Routine, Predictor, dataset and impacts_df
            if we_finetuned:
                self.we_finetuned_path = os.path.join(self.project_path, 'dataset_files', self.dataset_name,
                                                      self.model_name, 'sBERT')
            else:
                self.we_finetuned_path = str()

            self.routine, self.predictor = self.init_routine()

            _, _, self.word_relevance_df = self.routine.get_calculated_data('valid')
            # remove NaN from original dataset
            self.dataset = self.routine.valid_merged.copy().replace(pd.NA, '')

        else:
            # TODO: to complete, understand what to do with BERT to prepare the model for this dataframe
            self.dataset = dataset_df.replace(pd.NA, '')

        if self.dataset.shape[0] <= 0:
            raise ValueError(f'Dataset to evaluate must have some elements; the dataset has shape '
                             f'{self.dataset.shape[0]}.')

        self.match_df, self.match_ids, self.no_match_df, self.no_match_ids = self.get_match_no_match_df(
            delta=self.delta)

        if self.evaluate_positive:
            self.impacts_df = self.word_relevance_df[self.word_relevance_df.id.isin(self.match_ids)]
        else:
            self.impacts_df = self.word_relevance_df[self.word_relevance_df.id.isin(self.no_match_ids)]

        self.desc_to_eval_df = self.match_df if self.evaluate_positive else self.no_match_df

        # attributes defined at runtime during the evaluation of impacts_df
        self.ev = None
        self.prefix_wr = None

    def update_settings(self, dataset_name: str='', dataset_df: pandas.DataFrame=None, model_name: str= '', **kwargs):
        updated_dataset = False
        updated_model = False

        if dataset_name and dataset_df:
            raise ValueError("Specifying a dataset name is incorrect if a DataFrame is passed.")

        if (dataset_name and self.dataset_name != dataset_name) or (dataset_df and self.dataset != dataset_df):
            updated_dataset = True
            if dataset_name:
                self.dataset_name = dataset_name

            if dataset_df:
                self.dataset_name = str()

        if self.model_name and self.model_name != model_name:
            updated_model = True
            self.model_name = model_name

        if 'reset_files' in kwargs:
            self.reset_files = kwargs['reset_files']

        if 'reset_networks' in kwargs:
            self.reset_networks = kwargs['reset_networks']

        if 'we_finetuned' in kwargs:
            self.we_finetuned = kwargs['we_finetuned']

        if 'sentence_embeddings' in kwargs:
            self.sentence_embedding = kwargs['sentence_embedding']

        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']

        if 'additive_only' in kwargs:
            self.additive_only = kwargs['additive_only']

        if 'delta' in kwargs:
            self.delta = kwargs['delta']

        if updated_dataset or updated_model:
            if not dataset_df:
                self.model_files_path = os.path.join(self.project_path, 'dataset_files', dataset_name, self.model_name)
                self.init_routine()

                _, _, self.word_relevance_df = self.routine.get_calculated_data('valid')
                # remove NaN from original dataset
                self.dataset = self.routine.valid_merged.copy().replace(pd.NA, '')

            else:
                # TODO: to complete, understand what to do with BERT to prepare the model for this dataframe
                self.dataset = dataset_df

        if self.dataset.shape[0] <= 0:
            raise ValueError(f'Dataset to evaluate must have some elements; the dataset has shape '
                             f'{self.dataset.shape[0]}.')

        self.get_match_no_match_df(delta=self.delta)

        if self.evaluate_positive:
            self.impacts_df = self.word_relevance_df[self.word_relevance_df.id.isin(self.match_ids)]
        else:
            self.impacts_df = self.word_relevance_df[self.word_relevance_df.id.isin(self.no_match_ids)]

        self.desc_to_eval_df = self.match_df if self.evaluate_positive else self.no_match_df

        self.ev = None
        self.prefix_wr = None

    @staticmethod
    def get_prefix(em_df, word_relevance_df, side: str):
        assert side in ['left', 'right']
        word_relevance_el = word_relevance_df.copy().reset_index(drop=True)
        mapper = Mapper([x for x in em_df.columns if x.startswith(side + '_') and x != side + '_id'], r' ')
        available_prefixes = mapper.encode_attr(em_df).split()
        assigned_pref = list()
        word_prefixes = list()
        attr_to_code = {v: k for k, v in mapper.attr_map.items()}
        for i in range(word_relevance_el.shape[0]):
            word = str(word_relevance_el.loc[i, side + '_word'])
            if word == '[UNP]':
                word_prefixes.append('[UNP]')
            else:
                col = word_relevance_el.loc[i, side + '_attribute']
                col_code = attr_to_code[side + '_' + col]
                turn_prefixes = [x for x in available_prefixes if x[0] == col_code]
                idx = 0
                while idx < len(turn_prefixes) and word != turn_prefixes[idx][4:]:
                    idx += 1
                if idx < len(turn_prefixes):
                    tmp = turn_prefixes[idx]
                    del turn_prefixes[idx]
                    word_prefixes.append(tmp)
                    assigned_pref.append(tmp)
                else:
                    idx = 0
                    while idx < len(assigned_pref) and word != assigned_pref[idx][4:]:
                        idx += 1
                    if idx < len(assigned_pref):
                        word_prefixes.append(assigned_pref[idx])
                    else:
                        assert False, word

        return word_prefixes

    def append_prefix(self, exclude_attrs=('id', 'left_id', 'right_id', 'label'),
                      impact_col: str='token_contribution') -> pandas.DataFrame:

        if impact_col not in self.impacts_df.columns:
            raise ValueError(f"Missing column {impact_col} in dataframe.")

        ids = self.impacts_df['id'].unique()
        res = list()

        for id_ in ids:
            el = self.dataset[self.dataset.id == id_]
            word_relevance_el = self.impacts_df[self.impacts_df.id == id_]

            word_relevance_el['left_word_prefixes'] = self.get_prefix(el, word_relevance_el, 'left')
            word_relevance_el['right_word_prefixes'] = self.get_prefix(el, word_relevance_el, 'right')

            res.append(word_relevance_el.copy())

        res_df_ = pd.concat(res)

        mapper = Mapper(self.dataset.loc[:, np.setdiff1d(self.dataset.columns, exclude_attrs)], r' ')
        assert len(mapper.attr_map.keys()) % 2 == 0, 'The attributes must be the same for the two sources.'
        shift = int(len(mapper.attr_map.keys()) / 2)
        res_df_['right_word_prefixes'] = res_df_['right_word_prefixes'].apply(
            lambda x: chr(ord(x[0]) + shift) + x[1:] if x != '[UNP]' else x)

        if not self.evaluate_removing_du:
            df_list = list()
            res_df_[impact_col] = np.where((res_df_['left_word'] != '[UNP]') & (res_df_['right_word'] != '[UNP]'),
                                      res_df_[impact_col] / 2, res_df_[impact_col])

            for side in ['left', 'right']:
                side_columns = res_df_.columns[res_df_.columns.str.startswith(side)].to_list()
                token_impact = res_df_.loc[:, ['id', 'label', impact_col] + side_columns]
                token_impact.columns = token_impact.columns.str.replace(side + '_', '')
                token_impact = token_impact[token_impact['word'] != '[UNP]']
                token_impact['attribute'] = side + '_' + token_impact['attribute']

                df_list.append(token_impact)

            res_df_ = pd.concat(df_list, 0).reset_index(drop=True)
            res_df_ = res_df_.rename(columns={'word_prefixes': 'word_prefix'})

        res_df_ = res_df_.rename(columns={impact_col: 'impact'})

        if impact_col == 'pred':
            res_df_['impact'] -= 0.5

        self.impacts_df = res_df_

        return res_df_

    def get_match_no_match_df(self, delta=0.0):
        # match_df = df[pred > .5 + delta]
        match_df = self.dataset[self.dataset.label > .5 + delta]
        sample_len = min(100, match_df.shape[0])
        match_ids = match_df.id.sample(sample_len, random_state=0).values
        # no_match_df = df[(df['label'] < 0.5) & (pred >= pred_threshold)]
        no_match_df = self.dataset[self.dataset['label'] < 0.5]
        sample_len = min(100, no_match_df.shape[0])
        no_match_ids = no_match_df.id.sample(sample_len, random_state=0).values

        self.match_df = match_df
        self.match_ids = match_ids
        self.no_match_df = no_match_df
        self.no_match_ids = no_match_ids

        return match_df, match_ids, no_match_df, no_match_ids

    def evaluate_all_df(self, utility: Union[str, bool]='AOPC', explanation: str=''):
        for df_name in tqdm(list(self.sorted_dataset_names)):
            gc.collect()
            torch.cuda.empty_cache()
            self.evaluate_single_df(df_name, utility=utility, explanation=explanation)

    def evaluate_single_df(self, dataset_name: str='', utility: Union[str, bool]='AOPC', k: int=5,
                           explanation: str=''):

        if dataset_name and dataset_name != self.dataset_name:
            self.update_settings(dataset_name=dataset_name)

        ## from here ## means Andrea code that has been temporarily commented because not understood
        ## test_df = self.routine.test_merged.copy().replace(pd.NA, '')
        if explanation == 'LIME':
            ## pos_exp, neg_exp = self.load_explanations(turn_dataset_name=dataset_name, conf=explanation)
            ## match_ids = pos_exp.id.unique()
            ## no_match_ids = neg_exp.id.unique()

            if self.evaluate_removing_du:
                print("Warning: evaluation with LIME does not accept evaluating using Decision Unit. "
                      "Ignoring parameter evaluate_removing_du.")
            self.evaluate_removing_du = False

            ## res_df, ev_ = self.evaluate_df(test_df[test_df.id.isin(match_ids)], pos_exp, self.predictor, score_col='impact',
            ##                               utility=utility, k=k)
            ## self.calculate_save_metric(res_df, model_files_path, metric_name=utility, suffix=explanation)
            ##
            ## res_df, ev_ = self.evaluate_df(test_df[test_df.id.isin(no_match_ids)], neg_exp, self.predictor,
            ##                               score_col='impact', utility=utility, k=k)
            ## self.ev = ev_
            ## self.calculate_save_metric(res_df, model_files_path, metric_name=utility, prefix='no_match',
            ##                            suffix=explanation)
        else:
            # TODO: why df_name = test? what's going on with LIME?

            ## df_name = 'test'
            ## _, _, word_relevance = self.routine.get_calculated_data(df_name)
            ## df = self.routine.test_merged.copy().replace(pd.NA, '')
            if 'decision_unit_flat' in explanation:
                if self.evaluate_removing_du:
                    print("Warning: required explanation with decision_unit_flat, but evaluation with Decision Units is "
                          "required. Ignoring parameter evaluate_removing_du.")
                self.evaluate_removing_du = False

            elif 'remove_du_only' in explanation:
                if self.recompute_embeddings:
                    print("Warning: required explanation with remove_du_only, but it is required to recompute the word "
                          "embeddings. Ignoring parameter recompute_embeddings and updating the predictor.")
                self.recompute_embeddings = False

                ## predictor = partial(self.routine.get_predictor(recompute_embeddings=False), return_data=False, lr=True,
                ##                     chunk_size=self.batch_size, reload=True)
                self.predictor = partial(self.routine.get_predictor(recompute_embeddings=False), return_data=False, lr=True,
                                    chunk_size=self.batch_size, reload=True)

            ## match_df, match_ids, no_match_df, no_match_ids = self.get_match_no_match_df(df)

            # predictor = lambda x : [0.5]*x.shape[0]
            print('Evaluating data...')

            ## res_df, ev_ = self.evaluate_df(match_df[match_df.id.isin(match_ids)],
            ##                              word_relevance[word_relevance.id.isin(match_ids)], self.predictor,
            ##                              utility=utility, k=k)
            res_df, ev_ = self.evaluate_df(utility=utility, k=k)
            self.calculate_save_metric(res_df, self.model_files_path, metric_name=utility, suffix=explanation)

            ## if utility == 'AOPC':
            ##     return

            ## res_df, ev_ = self.evaluate_df(no_match_df[no_match_df.id.isin(no_match_ids)],
            ##                               word_relevance[word_relevance.id.isin(no_match_ids)], self.predictor,
            ##                               utility=utility, k=k)
            ## self.ev = ev_
            ## self.calculate_save_metric(res_df, model_files_path, metric_name=utility, prefix='no_match',
            ##                            suffix=explanation)

            return res_df, ev_

    def evaluate_df(self, exclude_attrs=('id', 'left_id', 'right_id', 'label'), score_col: str='token_contribution',
                    conf_name: str='bert', utility: Union[str, bool]='AOPC', k: int=5):

        ## ids = list(self.dataset.id.unique())
        print(f'Testing unit removal with -- {score_col}')

        self.append_prefix(exclude_attrs=exclude_attrs, impact_col=score_col)

        # EvaluateExplanation(em_df, word_relevance_prefix)
        ev_ = EvaluateExplanation(self.desc_to_eval_df, self.impacts_df, predict_method=self.predictor,
                                  exclude_attrs=exclude_attrs, percentage=.25, num_rounds=3,
                                  evaluate_removing_du=self.evaluate_removing_du,
                                  recompute_embeddings=self.recompute_embeddings,
                                  variable_side=self.variable_side, fixed_side=self.fixed_side)

        ## impacts_all = word_relevance_prefix
        ## impact_df = impacts_all[impacts_all.id.isin(ids)]
        ## start_el = em_df[em_df.id.isin(ids)]

        # variable_side = fixed_side = 'all'
        # evaluate_impacts(df_to_process, word_relevance_prefix)
        res = ev_.evaluate_impacts(add_before_perturbation=None, add_after_perturbation=None, utility=utility, k=k)
        res_df_ = pd.DataFrame(res)
        res_df_['conf'] = conf_name
        res_df_['error'] = res_df_.expected_delta - res_df_.detected_delta

        self.res_df = res_df_
        self.ev = ev_

        return res_df_, ev_

    def load_explanations(self, turn_dataset_name, conf='LIME'):
        base_files_path = os.path.join(self.project_path, 'dataset_files')
        turn_files_path = os.path.join(base_files_path, turn_dataset_name, 'wym')

        tmp_path = os.path.join(turn_files_path, f'negative_explanations_{conf}.csv')
        neg_exp = pd.read_csv(tmp_path)
        tmp_path = os.path.join(turn_files_path, f'positive_explanations_{conf}.csv')
        pos_exp = pd.read_csv(tmp_path)
        print('Loaded explanations')
        return pos_exp, neg_exp

    def init_routine(self, **kwargs):

        routine = Routine(self.dataset_name, self.dataset_path, self.project_path, reset_files=self.reset_files,
                          reset_networks=self.reset_networks, softlab_path=self.softlab_path, model_name=self.model_name,
                          we_finetuned=self.we_finetuned, we_finetune_path=self.we_finetuned_path,
                          sentence_embedding=self.sentence_embedding, device='cuda', **kwargs)

        routine.additive_only = self.additive_only
        routine.generate_df_embedding(chunk_size=self.batch_size)
        _ = routine.compute_word_pair()

        _ = routine.net_train(
            # batch_size=512, lr=5e-5
        )
        _ = routine.preprocess_word_pairs()
        _ = routine.EM_modelling(routine.features_dict, routine.words_pairs_dict, routine.train_merged,
                                 routine.test_merged,
                                 do_feature_selection=False)

        predictor = partial(routine.get_predictor(), return_data=False, lr=True, chunk_size=self.batch_size,
                            reload=True)

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

        print("---------- PERCHE' SONO QUA? -----------")
        model_name = 'BERT'
        model_files_path = os.path.join(self.project_path, 'dataset_files', dataset_name, model_name)
        routine, predictor = self.init_routine(verbose=True)
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


if __name__ == "__main__":
    # TODO: the problem could be in evaluate_removing_du. Check variable status for True or False in
    # TODO: method prepare_impacts of EvaluateExplanation
    wym_ev = WYMEvaluation('BeerAdvo-RateBeer', evaluate_removing_du=False, recompute_embeddings=True,
                           variable_side='left', fixed_side='right')
    results_df, ev_exp = wym_ev.evaluate_single_df(utility=True, explanation='last')
    print(results_df)

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
