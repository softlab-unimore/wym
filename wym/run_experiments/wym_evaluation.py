from typing import Union
import pandas as pd

from wym.import_utility import *
from wym.import_utility import prefix
from warnings import simplefilter
from tqdm.autonotebook import tqdm
# from Landmark_github.evaluation.Evaluate_explanation_Batch import *
from Landmark_github.evaluation.Evaluate_explanation_Batch import EvaluateExplanation, partial
from wym.BERTRoutine import Routine
from Landmark_github.landmark.landmark import Landmark, Mapper


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

        self.pos_res_df = None
        self.neg_res_df = None

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
            if not prefix:
                self.pos_res_df = res_df
            else:
                self.neg_res_df = res_df

            res_df['comb'] = np.where(res_df['comb_name'].str.startswith('top'), 'top', 'random')
            res_df['same_class'] = (res_df['start_pred'] > 0.5) == (res_df['new_pred'] > .5)
            sufficiency = res_df.groupby(['id', 'comb']).agg(
                {'same_class': [('sum', lambda x: x.sum() / x.size)]}).groupby('comb').mean()
            # assert False
            tmp_path = os.path.join(model_files_path, 'results', f'{prefix}{metric_name}_score{suffix}.csv')
            sufficiency.to_csv(tmp_path)
            print(sufficiency)


class WYMEvaluation(SoftlabEnv):
    def __init__(self, dataset_name: str = '', dataset_df: pd.DataFrame = None, model_name: str = 'BERT',
                 reset_files: bool = False, reset_networks: bool = False, we_finetuned: bool = True,
                 sentence_embedding: bool = False, batch_size: int = 256, delta: float = 0.1, additive_only: bool = False,
                 evaluate_removing_du: bool = True, recompute_embeddings: bool = True, variable_side: str = 'all',
                 fixed_side: str = 'all', score_col: str = 'token_contribution',
                 exclude_attrs: object = ('id', 'left_id', 'right_id', 'label'), add_before_perturbation: object = None,
                 add_after_perturbation: object = None, percentage: float = 0.25, num_rounds: int = 3):

        """
        Class to evaluate an Entity Matching dataset using WYM.


        :param dataset_name: string containing the name of the dataset which will load one of the available default
        datasets. Must be empty if dataset_df is passed.
        :param dataset_df: a pandas.DataFrame object containing the dataset to evaluate.
        :param model_name: the name of the underlying Deep Learning model to use for the creation of word embeddings.
        :param reset_files: specific BERTRoutine class parameter, resets the word embeddings files and recomputes
        them from scratch.
        :param reset_networks: specific BERTRoutine class parameter, resets the network cache and recomputes it.
        :param we_finetuned: specific BERTRoutine class parameter, allows usage of fine-tuned embeddings.
        :param sentence_embedding: specific BERTRoutine class parameter, uses sentence embeddings instead of
        word embeddings.
        :param batch_size: specific BERTRoutine class parameter, specifies the batch size for the generation of
        word embeddings.
        :param delta: value to use during the evaluation to decide when a new evaluation can be considered correct.
        The value is added to the default of 0.5.
        :param additive_only: specific BERTRoutine class parameter, specifies the model behavior for the computation
        of features.
        :param evaluate_removing_du: evaluates the dataset using Decision Units or single tokens.
        :param recompute_embeddings: specifies if the word embeddings must be recomputed after the perturbations or not.
        :param variable_side: the side of the Entity Matching dataset that can be perturbed.
        :param fixed_side: the side of the Entity Matching dataset that remains untouched.
        :param score_col: column name used to compute the score of the match during the evaluation.
        :param exclude_attrs: attributes to exclude from the evaluation, for example the "id" column.
        :param add_before_perturbation: TO_COMPLETE
        :param add_after_perturbation: TO_COMPLETE
        :param percentage: specific EvaluateExplanation class parameter, TO_COMPLETE
        :param num_rounds: specific EvaluateExplanation class parameter, TO_COMPLETE
        """
        if not dataset_name and not dataset_df:
            raise ValueError("Dataset name cannot be empty if the dataset DataFrame is None.")

        if dataset_name and dataset_df:
            raise ValueError("Specifying a dataset name is incorrect if a DataFrame is passed.")

        if not recompute_embeddings and not evaluate_removing_du:
            ValueError("It is not possible to evaluate on single tokens without recomputing embeddings.")

        if evaluate_removing_du and variable_side != 'all':
            ValueError("Invalid settings: required evaluation with Decision Units but the variable side is not 'all'.")

        super().__init__()
        # definition of WYM parameters
        self.project_path = os.path.join(self.softlab_path, 'Projects', 'WYM')
        self.evaluate_removing_du = evaluate_removing_du
        self.recompute_embeddings = recompute_embeddings
        self.dataset_name = dataset_name
        self.delta = delta

        # definition of evaluation parameters for the explanation
        self.fixed_accepted_sides = frozenset(['left', 'right', 'all', ''])
        self.variable_accepted_sides = frozenset(['left', 'right', 'all'])

        if variable_side == 'all' and fixed_side == 'all':
            fixed_side = str()

        if not variable_side and not fixed_side:
            raise ValueError("Invalid settings: variable side and fixed side cannot be both empty.")

        if variable_side not in self.variable_accepted_sides:
            raise ValueError("Invalid settings: variable side is not 'left', 'right' or 'all'.")

        if fixed_side not in self.fixed_accepted_sides:
            raise ValueError("Invalid settings: fixed side is not 'left', 'right', 'all' or empty.")

        if variable_side in ('left', 'right') and fixed_side == 'all':
            raise ValueError(f"Invalid settings: variable side is {variable_side} but fixed side is {fixed_side}.")

        if fixed_side in ('left', 'right') and variable_side == 'all':
            raise ValueError(f"Invalid settings: fixed side is {fixed_side} but variable side is {variable_side}.")

        if variable_side == fixed_side:
            raise ValueError(f"Invalid settings: variable and fixed sides are the same "
                             f"({variable_side}, {fixed_side}).")

        self.variable_side = variable_side
        self.fixed_side = fixed_side
        self.exclude_attrs = exclude_attrs
        self.score_col = score_col

        current_path_env = set(sys.path)  # speed up search for duplicates with set

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
            self.bert_params = {'reset_files', 'reset_networks', 'we_finetuned', 'sentence_embeddings', 'batch_size',
                                'additive_only'}

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

            _, _, self.word_relevance_df = self.routine.get_calculated_data('test')
            # remove NaN from original dataset
            self.dataset = self.routine.test_merged.copy().replace(pd.NA, '')

        else:
            # TODO: to complete, understand what to do with BERT to prepare the model for this dataframe
            self.dataset = dataset_df.replace(pd.NA, '')

        if self.dataset.shape[0] <= 0:
            raise ValueError(f'Dataset to evaluate must have some elements; the dataset has shape '
                             f'{self.dataset.shape[0]}.')

        self.match_df, self.match_ids, self.no_match_df, self.no_match_ids = self.get_match_no_match_df(
            delta=self.delta)

        self.pos_impacts_df = self.word_relevance_df[self.word_relevance_df.id.isin(self.match_ids)]
        self.neg_impacts_df = self.word_relevance_df[self.word_relevance_df.id.isin(self.no_match_ids)]

        if self.score_col not in self.pos_impacts_df.columns or self.score_col not in self.neg_impacts_df.columns:
            raise ValueError(f"Missing column {self.score_col} in dataframe.")

        self.append_prefix()

        # EvaluateExplanation(em_df, word_relevance_prefix)
        self.pos_ev = EvaluateExplanation(self.match_df, self.pos_impacts_df, predict_method=self.predictor,
                                          exclude_attrs=exclude_attrs, percentage=percentage,
                                          num_rounds=num_rounds, add_before_perturbation=add_before_perturbation,
                                          add_after_perturbation=add_after_perturbation,
                                          evaluate_removing_du=self.evaluate_removing_du,
                                          recompute_embeddings=self.recompute_embeddings,
                                          variable_side=self.variable_side, fixed_side=self.fixed_side)

        self.neg_ev = EvaluateExplanation(self.no_match_df, self.neg_impacts_df, predict_method=self.predictor,
                                          exclude_attrs=exclude_attrs, percentage=percentage,
                                          num_rounds=num_rounds, add_before_perturbation=add_before_perturbation,
                                          add_after_perturbation=add_after_perturbation,
                                          evaluate_removing_du=self.evaluate_removing_du,
                                          recompute_embeddings=self.recompute_embeddings,
                                          variable_side=self.variable_side, fixed_side=self.fixed_side)

        # attributes defined at runtime during the evaluation of impacts_df
        self.prefix_wr = None

    def update_settings(self, **kwargs):
        updated_dataset = False
        updated_model = False
        updated_evaluate_removing_du = False

        dataset_name = kwargs['dataset_name'] if 'dataset_name' in kwargs else str()
        dataset_df = kwargs['dataset_df'] if 'dataset_df' in kwargs else None

        if dataset_name and dataset_df:
            raise ValueError("Specifying a dataset name is incorrect if a DataFrame is passed.")

        if (dataset_name and self.dataset_name != dataset_name) or \
                (dataset_df is not None and
                 (not self.dataset.equals(dataset_df) or self.dataset.shape != dataset_df.shape)):
            updated_dataset = True

            if dataset_df is not None:
                self.dataset_name = str()

            if dataset_name:
                self.dataset_name = dataset_name

        model_name = kwargs['model_name'] if 'model_name' in kwargs else str()

        if model_name and self.model_name != model_name:
            updated_model = True
            self.model_name = model_name

        for bert_param in self.bert_params:
            if bert_param in kwargs:
                eval(f"self.{bert_param} = kwargs['{bert_param}']")

        if 'delta' in kwargs:
            self.delta = kwargs['delta']

        if 'exclude_attrs' in kwargs:
            self.exclude_attrs = kwargs['exclude_attrs']

        if 'score_col' in kwargs:
            self.score_col = kwargs['score_col']

        evaluate_removing_du = kwargs['evaluate_removing_du'] \
            if 'evaluate_removing_du' in kwargs else self.evaluate_removing_du
        recompute_embeddings = kwargs['recompute_embeddings'] \
            if 'recompute_embeddings' in kwargs else self.recompute_embeddings

        if not recompute_embeddings and not evaluate_removing_du:
            ValueError("Invalid settings: it is not possible to evaluate on single tokens without "
                       "recomputing embeddings.")
        else:
            if self.evaluate_removing_du != evaluate_removing_du:
                updated_evaluate_removing_du = True
                self.evaluate_removing_du = evaluate_removing_du

                if 'word_prefix' in self.pos_impacts_df:
                    self.pos_impacts_df = self.word_relevance_df[self.word_relevance_df.id.isin(self.match_ids)]

                if 'word_prefix' in self.neg_impacts_df:
                    self.neg_impacts_df = self.word_relevance_df[self.word_relevance_df.id.isin(self.no_match_ids)]

                if not (updated_dataset or updated_model):
                    self.append_prefix()

            self.recompute_embeddings = recompute_embeddings

        variable_side = kwargs['variable_side'] if 'variable_side' in kwargs else str()
        fixed_side = kwargs['fixed_side'] if 'fixed_side' in kwargs else str()

        if variable_side == 'all' and fixed_side == 'all':
            fixed_side = str()

        if variable_side or fixed_side:
            if not variable_side and fixed_side:
                raise ValueError(f"Invalid settings: variable side is empty but fixed side is {fixed_side}.")

            if variable_side:
                if variable_side not in self.variable_accepted_sides:
                    raise ValueError("Invalid settings: variable side is not left, right or all.")

                if (variable_side in ('left', 'right') and fixed_side == 'all') or \
                    (variable_side in ('left', 'right') and not fixed_side):
                    raise ValueError(
                        f"Invalid settings: variable side is {variable_side} but fixed side is empty.")

            if fixed_side:
                if fixed_side not in self.fixed_accepted_sides:
                    raise ValueError("Invalid settings: fixed side is not left, right, all or empty.")

                if (fixed_side in ('left', 'right') and variable_side == 'all') or \
                    (fixed_side in ('left', 'right') and not variable_side):
                    raise ValueError(
                        f"Invalid settings: fixed side is {fixed_side} but variable side is empty.")

            if variable_side == 'all' and fixed_side:
                print(f"Warning, invalid settings: variable side is {variable_side} and fixed side is not empty. "
                      f"Ignoring fixed side value.")
                fixed_side = str()

            if variable_side == fixed_side:
                raise ValueError(f"Invalid settings: variable and fixed sides are the same "
                                 f"({variable_side}, {fixed_side}).")

            if variable_side:
                self.variable_side = variable_side
                self.fixed_side = fixed_side

                kwargs['variable_side'] = variable_side
                kwargs['fixed_side'] = fixed_side

        if evaluate_removing_du and variable_side != 'all':
            ValueError("Invalid settings: required evaluation with Decision Units but the variable side is not 'all'.")

        if updated_evaluate_removing_du:
            kwargs['impacts_df'] = self.pos_impacts_df

        self.pos_ev.update_settings(**kwargs)

        if updated_evaluate_removing_du:
            kwargs['impacts_df'] = self.neg_impacts_df

        self.neg_ev.update_settings(**kwargs)

        if (updated_dataset or updated_model) and dataset_df is None:
            self.model_files_path = os.path.join(self.project_path, 'dataset_files', dataset_name, self.model_name)

            _, _, self.word_relevance_df = self.routine.get_calculated_data('test')
            # remove NaN from original dataset
            self.dataset = self.routine.valid_merged.copy().replace(pd.NA, '')

        elif dataset_df is not None:
            # TODO: to complete, understand what to do with BERT to prepare the model for this dataframe
            self.dataset = dataset_df.replace(pd.NA, '')

        if updated_dataset:
            if self.dataset.shape[0] <= 0:
                raise ValueError(f'Dataset to evaluate must have some elements; the dataset has shape '
                                 f'{self.dataset.shape[0]}.')

            self.init_routine()
            self.get_match_no_match_df(delta=self.delta)

            self.pos_impacts_df = self.word_relevance_df[self.word_relevance_df.id.isin(self.match_ids)]

            self.neg_impacts_df = self.word_relevance_df[self.word_relevance_df.id.isin(self.no_match_ids)]

            if self.score_col not in self.pos_impacts_df.columns or self.score_col not in self.neg_impacts_df.columns:
                raise ValueError(f"Missing column {self.score_col} in dataframe.")

            self.append_prefix()

            percentage = kwargs['percentage'] if 'percentage' in kwargs else 0.25
            num_rounds = kwargs['num_rounds'] if 'num_rounds' in kwargs else 3
            add_before_perturbation = kwargs['add_before_perturbation'] if 'add_before_perturbation' in kwargs else None
            add_after_perturbation = kwargs['add_after_perturbation'] if 'add_after_perturbation' in kwargs else None

            self.pos_ev = EvaluateExplanation(self.match_df, self.pos_impacts_df, predict_method=self.predictor,
                                          exclude_attrs=self.exclude_attrs, percentage=percentage,
                                          num_rounds=num_rounds, add_before_perturbation=add_before_perturbation,
                                          add_after_perturbation=add_after_perturbation,
                                          evaluate_removing_du=self.evaluate_removing_du,
                                          recompute_embeddings=self.recompute_embeddings,
                                          variable_side=self.variable_side, fixed_side=self.fixed_side)

            self.neg_ev = EvaluateExplanation(self.no_match_df, self.neg_impacts_df, predict_method=self.predictor,
                                              exclude_attrs=self.exclude_attrs, percentage=percentage,
                                              num_rounds=num_rounds, add_before_perturbation=add_before_perturbation,
                                              add_after_perturbation=add_after_perturbation,
                                              evaluate_removing_du=self.evaluate_removing_du,
                                              recompute_embeddings=self.recompute_embeddings,
                                              variable_side=self.variable_side, fixed_side=self.fixed_side)
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

    def append_prefix(self) -> pd.DataFrame:

        for impacts_df in (self.pos_impacts_df, self.neg_impacts_df):
            ids = impacts_df['id'].unique()
            res = list()
            compute_word_prefix_only = False

            if 'left_word_prefixes' in impacts_df.columns and 'right_word_prefixes' in impacts_df.columns:
                compute_word_prefix_only = True

            if not compute_word_prefix_only:
                for id_ in ids:
                    el = self.dataset[self.dataset.id == id_]
                    word_relevance_el = impacts_df[impacts_df.id == id_]

                    word_relevance_el['left_word_prefixes'] = self.get_prefix(el, word_relevance_el, 'left')
                    word_relevance_el['right_word_prefixes'] = self.get_prefix(el, word_relevance_el, 'right')

                    res.append(word_relevance_el.copy())

                res_df_ = pd.concat(res)

                if self.variable_side == 'all':
                    mapper = Mapper(self.dataset.loc[:, np.setdiff1d(self.dataset.columns, self.exclude_attrs)])
                    assert len(mapper.attr_map.keys()) % 2 == 0, 'The attributes must be the same for the two sources.'
                    shift = int(len(mapper.attr_map.keys()) / 2)
                    res_df_['right_word_prefixes'] = res_df_['right_word_prefixes'].apply(
                        lambda x: chr(ord(x[0]) + shift) + x[1:] if x != '[UNP]' else x)
            else: # the DataFrame doesn't need modifications, just compute the word_prefix column
                res_df_ = impacts_df

            if not self.evaluate_removing_du:
                df_list = list()

                score_col = 'impact' if compute_word_prefix_only else self.score_col

                res_df_[score_col] = np.where((res_df_['left_word'] != '[UNP]') & (res_df_['right_word'] != '[UNP]'),
                                                   res_df_[score_col] / 2, res_df_[score_col])

                for side in ['left', 'right']:
                    side_columns = res_df_.columns[res_df_.columns.str.startswith(side)].to_list()
                    token_impact = res_df_.loc[:, ['id', 'label', score_col] + side_columns]
                    token_impact.columns = token_impact.columns.str.replace(side + '_', '')
                    token_impact = token_impact[token_impact['word'] != '[UNP]']
                    token_impact['attribute'] = side + '_' + token_impact['attribute']

                    df_list.append(token_impact)

                res_df_ = pd.concat(df_list, 0).reset_index(drop=True)
                res_df_ = res_df_.rename(columns={'word_prefixes': 'word_prefix'})

            if not compute_word_prefix_only:
                res_df_ = res_df_.rename(columns={self.score_col: 'impact'})

            if self.score_col == 'pred':
                res_df_['impact'] -= 0.5

            if impacts_df.equals(self.pos_impacts_df):
                self.pos_impacts_df = res_df_

            else:
                self.neg_impacts_df = res_df_

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

    def evaluate_single_df(self, dataset_name: str='', dataset_df: pd.DataFrame=None, 
                           utility: Union[str, bool]='AOPC', k: int=5, explanation: str=''):

        if dataset_name and dataset_df:
            raise ValueError("Specifying a dataset name is incorrect if a DataFrame is passed.")

        if dataset_name and dataset_name != self.dataset_name:
            self.update_settings(dataset_name=dataset_name)

        if dataset_df:
            self.update_settings(dataset_df=dataset_df)

        if explanation == 'LIME':
            self.load_explanations(conf=explanation)

            if self.evaluate_removing_du:
                print("Warning: evaluation with LIME does not accept evaluating using Decision Unit. "
                      "Ignoring parameter evaluate_removing_du.")
            self.evaluate_removing_du = False

            self._evaluate_df(utility=utility, k=k)
            self.calculate_save_metric(self.pos_res_df, self.model_files_path, metric_name=utility, suffix=explanation)

            self._evaluate_df(utility=utility, k=k)
            self.calculate_save_metric(self.neg_res_df, self.model_files_path, metric_name=utility, prefix='no_match',
                                       suffix=explanation)
        else:
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

                self.predictor = partial(self.routine.get_predictor(recompute_embeddings=False), return_data=False, lr=True,
                                    chunk_size=self.batch_size, reload=True)

            print('Evaluating data.')

            self._evaluate_df(utility=utility, k=k)
            self.calculate_save_metric(self.pos_res_df, self.model_files_path, metric_name=utility, suffix=explanation)
            if self.neg_res_df is not None:
                self.calculate_save_metric(self.neg_res_df, self.model_files_path, metric_name=utility, prefix='no_match',
                                           suffix=explanation)

            return self.pos_res_df, self.pos_ev, self.neg_res_df, self.neg_ev

    def _evaluate_df(self, conf_name: str= 'bert', utility: Union[str, bool]= 'AOPC', k: int=5):

        ## ids = list(self.dataset.id.unique())
        print(f'Testing unit removal with -- {self.score_col}')

        print("Evaluating positive examples.")

        self.pos_res_df = self.pos_ev.evaluate_impacts(utility=utility, k=k)
        self.pos_res_df['conf'] = conf_name

        if utility == 'AOPC':  # applicable only on positive examples, don't evaluate negatives
            print("AOPC utility is applicable only on positive examples.")
            return self.pos_res_df, None

        print("Evaluating negative examples.")

        self.neg_res_df = self.neg_ev.evaluate_impacts(utility=utility, k=k)
        self.neg_res_df['conf'] = conf_name

        return self.pos_res_df, self.neg_res_df

    def load_explanations(self, conf='LIME'):
        neg_exp_path = os.path.join(self.model_files_path, f'negative_explanations_{conf}.csv')
        pos_exp_path = os.path.join(self.model_files_path, f'positive_explanations_{conf}.csv')
        neg_exp = pd.read_csv(neg_exp_path)
        pos_exp = pd.read_csv(pos_exp_path)
        print('Loaded explanations')

        self.match_ids = pos_exp.id.unique()
        self.no_match_ids = neg_exp.id.unique()
        self.match_df = self.dataset[self.dataset.id.isin(self.match_ids)]
        self.no_match_df = self.dataset[self.dataset.id.isin(self.no_match_ids)]

        self.pos_ev.update_settings(impacts_df=self.match_df)
        self.neg_ev.update_settings(impacts_df=self.no_match_df)

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
    utility = True
    variable_side='all'
    fixed_side='all'
    explanation = 'LIME'
    evaluate_removing_du = True
    wym_ev = WYMEvaluation('BeerAdvo-RateBeer', evaluate_removing_du=evaluate_removing_du, recompute_embeddings=True,
                           variable_side=variable_side, fixed_side=fixed_side)

    pos_results_df, pos_ev_expl, neg_results_df, neg_ev_expl = wym_ev.evaluate_single_df(utility=utility,
                                                                                         explanation='last')

    # wym_ev = WYMEvaluation('BeerAdvo-RateBeer', evaluate_removing_du=True, recompute_embeddings=True,
    #                        variable_side='all')
    # pos_results_df, pos_ev_expl, neg_results_df, neg_ev_expl = wym_ev.evaluate_single_df(utility=utility,
    #                                                                                      explanation=explanation)
    print(pos_results_df)

    pos_ev_expl.generate_counterfactual_examples()

    if not os.path.exists('./evaluation_tables'):
        os.makedirs("./evaluation_tables")

    with open(f"./evaluation_tables/counterfactuals_{wym_ev.dataset_name}_utility_{utility}_exp_{explanation}.html",
              'w') as html_file:
        html_file.write(pos_ev_expl.plot_counterfactual())
