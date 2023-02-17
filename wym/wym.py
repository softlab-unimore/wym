import copy
import os
import pickle
from warnings import simplefilter

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm

import numpy as np
import pandas as pd
import torch

from .FeatureContribution import FeatureContribution
from .FeatureExtractor import FeatureExtractor
from .Net import DatasetAccoppiate, NetAccoppiate, train_model
from .WordEmbedding import WordEmbedding
from .WordPairGenerator import WordPairGenerator


class Wym:
    def __init__(self, df: pd.DataFrame, we_finetune_path='bert-base-uncased', device='auto',
                 exclude_attrs=['id', 'left_id', 'right_id', 'label'],
                 column_prefixes=['left_', 'right_'], reset_networks=False, model_files_path='wym',
                 batch_size=256, verbose=True):
        self.columns_to_use = np.setdiff1d(df.columns, exclude_attrs)
        self.cols = pd.Series(self.columns_to_use.copy())
        self.cols = self.cols[self.cols.str.startswith(column_prefixes[0])].str.replace(column_prefixes[0], '')
        self.lp = column_prefixes[0]
        self.rp = column_prefixes[1]
        if device is None or device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model_files_path = model_files_path
        os.makedirs(self.model_files_path, exist_ok=True)
        self.reset_networks = reset_networks
        self.batch_size = batch_size
        self.verbose = verbose
        self.additive_only = False

        # simplified Word Embedding interface
        self.we = WordEmbedding(device=self.device, verbose=True, model_path=we_finetune_path)
        self.feature_extractor = FeatureExtractor()

    @staticmethod
    def convert_token_contribution_to_impact(du_contributions: np.array, bias: np.array) -> np.array:
        e = np.exp
        softmax = lambda x: e(x) / sum(e(x))

        all_contrib = np.concatenate([du_contributions, bias])
        impacts = np.zeros_like(all_contrib)

        pos_mask = all_contrib >= 0

        pos_contrib = all_contrib[pos_mask]
        neg_contrib = all_contrib[~pos_mask]

        pos_sum = pos_contrib.sum()
        neg_sum = neg_contrib.sum()

        tot_sum = pos_sum + neg_sum

        pred_pos = ((e(tot_sum) * pos_sum) /
                    ((e(tot_sum) + 1) * tot_sum))

        pred_neg = ((e(tot_sum) * neg_sum) /
                    ((e(tot_sum) + 1) * tot_sum))

        impacts[pos_mask] = softmax(pos_contrib) * pred_pos
        impacts[~pos_mask] = softmax(neg_contrib) * pred_neg

        return impacts

    def split_X_y(self, df, label_column_name='label'):
        return df[self.columns_to_use], df[label_column_name]

    def get_processed_data(self, df, batch_size=None, verbose=False):
        if hasattr(self, 'batch_size'):
            batch_size = self.batch_size
        we = self.we
        res = {}
        for side in ['left', 'right']:
            if verbose:
                print(f'Embedding {side} side')
            prefix = self.lp if side == 'left' else self.rp
            cols = [prefix + col for col in self.cols]
            tmp_df = df.loc[:, cols]
            res[side + '_word_map'] = WordPairGenerator.map_word_to_attr(tmp_df, self.cols, prefix=prefix)
            emb, words = we.generate_embedding(tmp_df, chunk_size=batch_size)

            res[side + '_emb'] = emb
            res[side + '_words'] = words
        return res

    def get_word_pairs(self, df, data_dict, use_schema=True, **kwargs):
        wp = WordPairGenerator(df=df, use_schema=use_schema, device=self.device, verbose=self.verbose,
                               **kwargs)
        res = wp.get_word_pairs(df, data_dict)
        word_pairs, emb_pairs = res
        word_pairs = pd.DataFrame(word_pairs)
        return word_pairs, emb_pairs

    def net_train(self, train_word_pairs=None, train_emb_pairs=None, valid_word_pairs=None, valid_emb_pairs=None,
                  num_epochs=40, lr=3e-5, batch_size=256, ):

        data_loader = DatasetAccoppiate(train_word_pairs, train_emb_pairs)
        self.train_data_loader = data_loader
        best_model = NetAccoppiate()
        device = self.device
        tmp_path = os.path.join(self.model_files_path, 'net.pickle')
        try:
            assert self.reset_networks == False, 'resetting networks'
            best_model.load_state_dict(torch.load(tmp_path, map_location=torch.device(device)))
        except Exception as e:
            print(e)
            net = NetAccoppiate()
            net.to(device)
            criterion = nn.BCELoss().to(device)
            # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=.9)
            optimizer = optim.Adam(net.parameters(), lr=lr)

            train_dataset = data_loader
            valid_dataset = copy.deepcopy(train_dataset)
            valid_dataset.__init__(valid_word_pairs, valid_emb_pairs)

            dataloaders_dict = {'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
                                'valid': DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)}

            best_model, score_history, last_model = train_model(net,
                                                                dataloaders_dict, criterion, optimizer,
                                                                nn.MSELoss().to(device), num_epochs=num_epochs,
                                                                device=device)
            # optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=.9)
            # best_model, score_history, last_model = train_model(net,dataloaders_dict, criterion, optimizer,nn.MSELoss().to(device), num_epochs=150, device=device)

            out = net(valid_dataset.X.to(device))
            print(f'best_valid --> mean:{out.mean():.4f}  std: {out.std():.4f}')
            out = last_model(valid_dataset.X.to(device))
            print(f'last_model --> mean:{out.mean():.4f}  std: {out.std():.4f}')
            print('Save...')
            torch.save(best_model.state_dict(), tmp_path)

        self.word_pair_model = best_model
        return best_model

    def relevance_score(self, word_pairs, emb_pairs):
        self.word_pair_model.eval()
        self.word_pair_model.to(self.device)
        data_loader = self.train_data_loader
        data_loader.__init__(word_pairs, emb_pairs)
        word_pair_corrected = data_loader.word_pairs_corrected
        with torch.no_grad():
            word_pair_corrected['pred'] = self.word_pair_model(data_loader.X.to(self.device)).cpu().detach().numpy()
        return word_pair_corrected

    def extract_features(self, word_pair, **kwargs):
        features = self.feature_extractor.extract_features_by_attr(word_pair, self.cols, **kwargs)
        return features

    def EM_modelling(self, X_train, y_train, X_valid, y_valid,
                     do_evaluation=False, do_feature_selection=False, results_path='results'):
        if self.additive_only and results_path == 'results':
            results_path = 'results_additive'
        if not hasattr(self, 'models'):
            mmScaler = MinMaxScaler()
            mmScaler.clip = False
            self.models = [
                ('LR',
                 Pipeline([('mm', copy.copy(mmScaler)), ('LR', LogisticRegression(max_iter=200, random_state=0))])),
                ('LR_std',
                 Pipeline(
                     [('mm', copy.copy(StandardScaler())), ('LR', LogisticRegression(max_iter=200, random_state=0))])),
                # ('LDA', Pipeline([('mm', copy.copy(mmScaler)), ('LDA', LinearDiscriminantAnalysis())])),
                # ('LDA_std', Pipeline([('mm', copy.copy(StandardScaler())), ('LDA', LinearDiscriminantAnalysis())])),
                # ('KNN', Pipeline([('mm', copy.copy(mmScaler)), ('KNN', KNeighborsClassifier())])),
                # ('CART', DecisionTreeClassifier(random_state=0)),
                # ('NB', GaussianNB()),
                # ('SVM', Pipeline([('mm', copy.copy(mmScaler)), ('SVM', SVC(probability=True, random_state=0))])),
                # ('AB', AdaBoostClassifier(random_state=0)),
                # ('GBM', GradientBoostingClassifier(random_state=0)),
                # ('RF', RandomForestClassifier(random_state=0)),
                # ('ET', ExtraTreesClassifier(random_state=0)),
                # ('dummy', DummyClassifier(strategy='stratified', random_state=0)),
            ]
        # models.append(('Vote', VotingClassifier(models[:-1], voting='soft')))
        model_names = [x[0] for x in self.models]

        res = {(x, y): [] for x in ['train', 'valid'] for y in ['f1', 'precision', 'recall']}
        df_pred = {}
        time_list_dict = []
        time_dict = {}
        for name, model in tqdm(self.models):
            model.fit(X_train.values, y_train)

            for turn_name, turn_df, turn_y in zip(['train', 'valid'], [X_train, X_valid],
                                                  [y_train, y_valid]):
                df_pred[turn_name] = model.predict(turn_df)
                for score_name, scorer in [['f1', f1_score], ['precision', precision_score], ['recall', recall_score]]:
                    score_value = scorer(turn_y, df_pred[turn_name])
                    res[(turn_name, score_name)].append(score_value)
                    # if turn_name == 'test' and score_name == 'f1':
                    #     print(f'{name:<10}-{score_name} {score_value}')

        print('before feature selection')
        res_df = pd.DataFrame(res, index=model_names)
        res_df.index.name = 'model_name'
        best_f1 = res_df[('valid', 'f1')].max()
        best_features = X_train.columns
        best_model_name = res_df.iloc[[res_df[('valid', 'f1')].argmax()]].index.values[0]
        for x in self.models:
            if x[0] == best_model_name:
                best_model = x[1]

        best_model.fit(X_train.values, y_train)
        model_data = {'features': best_features, 'model': best_model}
        tmp_path = os.path.join(self.model_files_path, 'best_feature_model_data.pickle')
        self.best_model_data = model_data
        with open(tmp_path, 'wb') as file:
            pickle.dump(model_data, file)

        mask = (res_df[('valid', 'f1')].argmax()) & (res_df.index.str.contains('LR'))
        best_linear_model_name = res_df.iloc[mask].index.values[0]
        for x in self.models:
            if x[0] == best_linear_model_name:
                best_linear_model = x[1]
            # linear_model = Pipeline([('LR', LogisticRegression(max_iter=200, random_state=0))])
        # # LogisticRegression(max_iter=200, random_state=0)
        # linear_model.fit(X_train.values, y_train)
        model_data = {'features': best_features, 'model': best_linear_model}
        self.best_linear_model_data = model_data
        tmp_path = os.path.join(self.model_files_path, 'linear_model.pickle')
        with open(tmp_path, 'wb') as file:
            pickle.dump(model_data, file)

    def get_match_score(self, features_df, lr=False, reload=False):
        self.load_model(lr=lr, reload=reload)
        X = features_df[self.best_features].to_numpy()
        if isinstance(self.feature_model, Pipeline) and isinstance(self.feature_model[0], MinMaxScaler):
            self.feature_model[0].clip = False
        match_score = self.feature_model.predict_proba(X)[:, 1]
        match_score_series = pd.Series(0.5, index=features_df.index)
        features = features_df
        match_score_series[features.index] = match_score
        match_score = match_score_series.values
        return match_score

    def load_model(self, lr=False, reload=False):
        if lr is True:
            if not hasattr(self, 'best_linear_model_data') or reload:
                tmp_path = os.path.join(self.model_files_path, 'linear_model.pickle')
                with open(tmp_path, 'rb') as file:
                    model_data = pickle.load(file)
            else:
                model_data = self.best_linear_model_data
        else:
            if not hasattr(self, 'best_model_data') or reload:
                tmp_path = os.path.join(self.model_files_path, 'best_feature_model_data.pickle')
                with open(tmp_path, 'rb') as file:
                    model_data = pickle.load(file)
            else:
                model_data = self.best_model_data
        self.feature_model = model_data['model']
        self.best_features = model_data['features']

    def fit(self, X: pd.DataFrame, y, valid_X=None, valid_y=None):
        X = X.copy()
        X['label'] = y
        res_dict = self.get_processed_data(X, batch_size=self.batch_size)
        word_pairs, emb_pairs = self.get_word_pairs(X, res_dict)

        if valid_X is None or valid_y is None:
            valid_X, valid_y = X.copy(), y.copy()
            valid_res_dict, valid_word_pairs, valid_emb_pairs = res_dict, word_pairs, emb_pairs
        else:
            valid_res_dict = self.get_processed_data(valid_X)
            valid_word_pairs, valid_emb_pairs = self.get_word_pairs(valid_X, valid_res_dict)

        _ = self.net_train(train_word_pairs=word_pairs,
                           train_emb_pairs=emb_pairs,
                           valid_word_pairs=valid_word_pairs,
                           valid_emb_pairs=valid_emb_pairs)
        word_relevance = self.relevance_score(word_pairs, emb_pairs)
        features = self.extract_features(word_relevance)

        valid_word_relevance = self.relevance_score(valid_word_pairs, valid_emb_pairs)
        valid_features = self.extract_features(valid_word_relevance)

        self.EM_modelling(X_train=features, y_train=y, X_valid=valid_features, y_valid=valid_y)

    def predict(self, X, lr=True, reload=False, return_data=False):
        df_to_process = X.copy()
        if 'id' not in df_to_process.columns:
            df_to_process = df_to_process.reset_index(drop=True)
            df_to_process['id'] = df_to_process.index

        data_dict = self.get_processed_data(df_to_process)
        word_pairs, emb_pairs = self.get_word_pairs(X, data_dict)
        word_relevance = self.relevance_score(word_pairs, emb_pairs)
        features = self.extract_features(word_relevance)

        match_score = self.get_match_score(features, lr=lr, reload=reload)
        match_score_series = pd.Series(0.5, index=df_to_process.id)
        match_score_series[features.index] = match_score
        match_score = match_score_series.values

        if return_data:
            if lr:
                lr = self.best_linear_model_data['model']['LR']
                co_df = pd.Series(lr.coef_.squeeze(), index=features.columns)
                turn_contrib = FeatureContribution.extract_features_by_attr(word_relevance,
                                                                            self.cols)  # no additive_only param

                data = turn_contrib.loc[:, features.columns].dot(co_df)
                word_relevance['token_contribution'] = data
            return match_score, data_dict, word_pairs, emb_pairs, features, word_relevance
        else:
            return match_score

    @staticmethod
    def plot_token_contribution(el_df, score_col='token_contribution', cut=0.1):
        plt.rcParams.update({'font.size': 8, "figure.figsize": (18, 6)})

        tmp_df = el_df.copy()
        tmp_df = tmp_df.set_index(['left_word', 'right_word'])
        tmp_df = tmp_df[tmp_df[score_col].abs() >= cut]
        # colors = ['orange' ] * tmp_df.shape[0]
        colors = np.where(tmp_df[score_col] >= 0, 'green', 'red')

        g = tmp_df.plot(y=score_col, kind='barh', color=colors, alpha=.5, legend='')

        for p in g.patches:
            width = p.get_width()
            offset = 35
            offset = offset if width > 0 else -offset
            g.annotate(format(width, '.3f'),
                       (width, p.get_y()),
                       ha='right' if width > 0 else 'left',
                       va='center',
                       xytext=(offset, 2),
                       textcoords='offset points')

            plt.ylabel('')

        g.grid(axis='y', color='0.7')
        g.axes.set_yticklabels(g.axes.get_yticklabels(), fontsize=9)
        yticks = g.get_yticks()
        for y0, y1 in zip(yticks[::2], yticks[1::2]):
            g.axhspan(y0, y1, color='0.7', alpha=0.1, zorder=0)
        g.set_yticks(yticks)  # force the same yticks again
        plt.tight_layout()
        plt.show()

    def generate_counterfactual(self, desc_df) -> pd.DataFrame:
        match_score, data_dict, word_pairs, emb_pairs, features, word_relevance = self.predict(X_test, return_data=True)



if __name__ == '__main__':
    simplefilter(action='ignore', category=FutureWarning)
    dataset_path = '/home/baraldian/Abt-Buy/'

    simplified_columns = ['id', 'left_id', 'right_id', 'label', 'left_name', 'right_name']
    train_df = pd.read_csv(dataset_path + 'train_merged.csv')[simplified_columns].iloc[:500]
    valid_df = pd.read_csv(dataset_path + 'valid_merged.csv')[simplified_columns].iloc[:500]
    test_df = pd.read_csv(dataset_path + 'test_merged.csv')[simplified_columns].iloc[:500]

    # train_df.to_csv(dataset_path + 'train_simplified.csv', index=False)
    # valid_df.to_csv(dataset_path + 'valid_simplified.csv', index=False)
    # test_df.to_csv(dataset_path + 'test_simplified.csv', index=False)

    exclude_attrs = ['id', 'left_id', 'right_id', 'label']
    wym = Wym(df=train_df, exclude_attrs=exclude_attrs, batch_size=1600)
    X, y = train_df[wym.columns_to_use], train_df['label']
    X_valid, y_valid = valid_df[wym.columns_to_use], valid_df['label']
    X_test, y_test = test_df[wym.columns_to_use], test_df['label']
    wym.fit(X, y, X_valid, y_valid)
    wym.predict(X_test)