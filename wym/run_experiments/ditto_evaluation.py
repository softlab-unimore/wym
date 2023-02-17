import os
import sys
from warnings import simplefilter

from tqdm.autonotebook import tqdm
from wym_github.wym.run_experiments.wym_evaluation import SoftlabEnv


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