import copy
import gc
import os
import pickle
import sys
from warnings import simplefilter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from FeatureExtractor import FeatureExtractor
from Net import DatasetAccoppiate, NetAccoppiate, train_model
from WordEmbedding import WordEmbedding
from WordPairGenerator import WordPairGenerator


class Routine():
    def __init__(self, dataset_name, dataset_path, project_path,
                 reset_files=False, model_name='BERT', device=None, reset_networks=False, clean_special_char=True,
                 col_to_drop=['left_price', 'right_price'],
                 softlab_path='/content/drive/Shareddrives/SoftLab/'):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        simplefilter(action='ignore', category=FutureWarning)
        simplefilter(action='ignore')
        pd.options.display.float_format = '{:.4f}'.format
        self.softlab_path = os.path.join(softlab_path)
        self.softlab_path = os.path.join(softlab_path)
        self.reset_files = reset_files  # @ param {type:"boolean"}
        self.reset_networks = reset_networks  # @ param {type:"boolean"}
        self.dataset_name = dataset_name
        self.model_name = model_name
        if dataset_path == None:
            self.dataset_path = os.path.join(softlab_path, 'Dataset', 'Entity Matching', dataset_name)
        self.project_path = os.path.join(softlab_path, 'Projects', 'Concept level EM (exclusive-inclluse words)')
        self.model_files_path = os.path.join(self.project_path, 'dataset_files', dataset_name, model_name)
        try:
            os.makedirs(self.model_files_path)
        except:
            pass
        try:
            os.makedirs(os.path.join(self.model_files_path, 'results'))
        except:
            pass

        sys.path.append(os.path.join(project_path, 'common_functions'))
        sys.path.append(os.path.join(project_path, 'src'))
        pd.options.display.max_colwidth = 130
        self.col_to_drop = col_to_drop
        self.train = pd.read_csv(os.path.join(dataset_path, 'train_merged.csv')).drop(self.col_to_drop, 1)
        self.test = pd.read_csv(os.path.join(dataset_path, 'test_merged.csv')).drop(self.col_to_drop, 1)
        self.valid = pd.read_csv(os.path.join(dataset_path, 'valid_merged.csv')).drop(self.col_to_drop, 1)
        self.table_A = pd.read_csv(os.path.join(dataset_path, 'tableA.csv'))
        self.table_B = pd.read_csv(os.path.join(dataset_path, 'tableB.csv'))
        self.table_A_orig = self.table_A.copy()

        left_ids = []
        right_ids = []
        for df in [self.train, self.valid, self.test]:
            left_ids.append(df.left_id.values)
            right_ids.append(df.right_id.values)
        left_ids = np.unique(np.concatenate(left_ids))
        right_ids = np.unique(np.concatenate(right_ids))
        self.table_A[~self.table_A.id.isin(left_ids)] = None
        self.table_B[~self.table_B.id.isin(right_ids)] = None
        self.cols = np.setdiff1d(self.table_A.columns, ['id'])
        if clean_special_char:
            spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")",
                          "*", "+", ",", "-", "/", ":", ";", "<",
                          "=", ">", "?", "@", "[", "\\", "]", "^", "_",
                          "`", "{", "|", "}", "~", "–", "´"]

            for col in np.setdiff1d(self.table_A.columns, ['id']):
                self.table_A[col] = self.table_A[col].astype(str). \
                                        str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode(
                    'utf-8') + ' '
                self.table_B[col] = self.table_B[col].astype(str). \
                                        str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode(
                    'utf-8') + ' '
                for char in spec_chars:
                    self.table_A[col] = self.table_A[col].str.replace(' \\' + char + ' ', ' ')
                    self.table_B[col] = self.table_B[col].str.replace(' \\' + char + ' ', ' ')

                self.table_A[col] = self.table_A[col].str.replace('-', ' ')
                self.table_B[col] = self.table_B[col].str.replace('-', ' ')
                self.table_A[col] = self.table_A[col].str.split().str.join(" ")
                self.table_B[col] = self.table_B[col].str.split().str.join(" ")

        self.table_A = self.table_A.replace('None', np.nan).replace('nan', np.nan)
        self.table_B = self.table_B.replace('None', np.nan).replace('nan', np.nan)
        self.words_divided = {}
        for name, df in zip(['table_A', 'table_B'], [self.table_A, self.table_B]):
            tmp_res = []
            for i in tqdm(range(df.shape[0])):
                el = df.iloc[[i]]
                el_words = {}
                for col in self.cols:
                    if el[col].notna().values[0]:
                        el_words[col] = str(el[col].values[0]).split()
                    else:
                        el_words[col] = []
                tmp_res.append(el_words.copy())
            self.words_divided[name] = tmp_res

    def generate_df_embedding(self, chunk_size=1000):
        self.embeddings = {}
        self.words = {}
        try:
            assert self.reset_files == False, 'Reset_files'
            for df_name in ['table_A', 'table_B']:
                tmp_path = os.path.join(self.model_files_path, 'emb_' + df_name + '.csv')
                with open(tmp_path, 'rb') as file:
                    self.embeddings[df_name] = torch.load(file)
                tmp_path = os.path.join(self.model_files_path, 'words_list_' + df_name + '.csv')
                with open(tmp_path, 'rb') as file:
                    self.words[df_name] = pickle.load(file)
            print('Loaded ')
        except Exception as e:
            print(e)
            we = WordEmbedding(device=self.device)
            for name, df in [('table_A', self.table_A), ('table_B', self.table_B)]:
                gc.collect()
                torch.cuda.empty_cache()
                emb, words = we.generate_embedding(df, chunk_size=chunk_size)
                self.embeddings[name] = emb
                self.words[name] = words
                tmp_path = os.path.join(self.model_files_path, 'emb_' + name + '.csv')
                with open(tmp_path, 'wb') as file:
                    torch.save(emb, file)
                tmp_path = os.path.join(self.model_files_path, 'words_list_' + name + '.csv')
                with open(tmp_path, 'wb') as file:
                    pickle.dump(words, file)

    def compute_word_pair(self, use_schema=True):
        we = WordEmbedding(device=self.device)
        words_pairs_dict, emb_pairs_dict = {}, {}
        try:
            assert self.reset_files == False, 'Reset_files'
            for df_name in ['train', 'valid', 'test']:
                tmp_path = os.path.join(self.model_files_path, df_name + 'word_pairs.csv')
                words_pairs_dict[df_name] = pd.read_csv(tmp_path)

                tmp_path = os.path.join(self.model_files_path, df_name + 'emb_pairs.csv')
                with open(tmp_path, 'rb') as file:
                    emb_pairs_dict[df_name] = pickle.load(file)
            print('Loaded ')
        except Exception as e:
            print(e)

            word_pair_generator = WordPairGenerator(self.words, self.embeddings, self.words_divided, df=self.test,
                                                    use_schema=use_schema)
            for df_name, df in zip(['train', 'valid', 'test'], [self.train, self.valid, self.test]):
                word_pairs, emb_pairs = word_pair_generator.process_df(df)
                tmp_path = os.path.join(self.model_files_path, df_name + 'word_pairs.csv')
                words_pairs_dict[df_name] = pd.DataFrame(word_pairs)
                words_pairs_dict[df_name].to_csv(tmp_path, index=False)

                tmp_path = os.path.join(self.model_files_path, df_name + 'emb_pairs.csv')
                with open(tmp_path, 'wb') as file:
                    pickle.dump(emb_pairs, file)
                emb_pairs_dict[df_name] = emb_pairs

        self.words_pairs_dict = words_pairs_dict
        self.emb_pairs_dict = emb_pairs_dict
        return words_pairs_dict, emb_pairs_dict

    def net_train(self, num_epochs=100, lr=0.00001, batch_size=128):
        word_pairs = self.words_pairs_dict['train'].copy()
        emb_pairs = self.emb_pairs_dict['train']
        data_loader = DatasetAccoppiate(word_pairs, emb_pairs)
        self.train_data_loader = data_loader
        best_model = NetAccoppiate()
        device = self.device
        tmp_path = os.path.join(self.model_files_path, 'net0.pickle')
        try:
            assert self.reset_networks == False, 'resetting networks'
            best_model.load_state_dict(torch.load(tmp_path,
                                                  map_location=torch.device(device)))
        except Exception as e:
            print(e)
            net = NetAccoppiate()
            net.to(device)
            criterion = nn.BCELoss().to(device)
            # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=.9)
            optimizer = optim.Adam(net.parameters(), lr=lr)

            train_dataset = data_loader
            valid_dataset = copy.deepcopy(train_dataset)
            valid_dataset.__init__(self.words_pairs_dict['valid'], self.emb_pairs_dict['valid'])

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

    def preprocess_word_pairs(self):
        processor = WordsPairsProcessor(self.train)

        features_dict = {}
        word_pair_dict = {}
        for name in ['train', 'valid', 'test']:
            feat, word_pairs = processor.extract_features(self.word_pair_model, self.words_pairs_dict[name],
                                                          self.emb_pairs_dict[name], self.train_data_loader)
            features_dict[name] = feat
            word_pair_dict[name] = word_pairs
        self.features_dict, self.word_pair_dict = features_dict, word_pair_dict
        return features_dict, word_pair_dict

    def EM_modelling(self, features_dict, word_pair_dict, train, test):

        models = [('LR', Pipeline([('mm', MinMaxScaler()), ('LR', LogisticRegression(max_iter=200, random_state=0))])),
                  ('LDA', Pipeline([('mm', MinMaxScaler()), ('LDA', LinearDiscriminantAnalysis())])),
                  ('KNN', Pipeline([('mm', MinMaxScaler()), ('KNN', KNeighborsClassifier())])),
                  ('CART', DecisionTreeClassifier(random_state=0)),
                  ('NB', GaussianNB()),
                  ('SVM', Pipeline([('mm', MinMaxScaler()), ('SVM', SVC(probability=True, random_state=0))])),
                  ('AB', AdaBoostClassifier(random_state=0)),
                  ('GBM', GradientBoostingClassifier(random_state=0)),
                  ('RF', RandomForestClassifier(random_state=0)),
                  ('ET', ExtraTreesClassifier(random_state=0)),
                  ('dummy', DummyClassifier(strategy='stratified', random_state=0)),
                  ]
        models.append(('Vote', VotingClassifier(models[:-1], voting='soft')))
        model_names = [x[0] for x in models]

        X_train, y_train = self.features_dict['train'].to_numpy(), self.train.label
        X_valid, y_valid = self.features_dict['valid'].to_numpy(), self.valid.label
        X_test, y_test = self.features_dict['test'].to_numpy(), self.test.label

        res = {(x, y): [] for x in ['train', 'test'] for y in ['f1', 'precision', 'recall']}
        for name, model in tqdm(models):
            model.fit(X_train, y_train)
            for score_name, scorer in [['f1', f1_score], ['precision', precision_score], ['recall', recall_score]]:
                res[('train', score_name)].append(scorer(y_train, model.predict(X_train)))
                res[('test', score_name)].append(scorer(y_test, model.predict(X_test)))
        self.models = models
        pd.options.display.float_format = '{:.4f}'.format

        res_df = pd.DataFrame(res, index=model_names)
        res_df.index.name = 'model_name'
        res_df.to_csv(os.path.join(self.model_files_path, 'results', 'performances.csv'))
        return res_df

    def plot_rf(self, rf, columns):
        pd.DataFrame([rf.feature_importances_], columns=columns).T.plot.bar(figsize=(25, 5));


class EMFeatures:
    def __init__(self, df, exclude_attrs=['id', 'left_id', 'right_id', 'label'], device=None, n_proc=1):
        self.n_proc = n_proc
        self.cols = [x[5:] for x in df.columns if x not in exclude_attrs and x.startswith('left_')]
        self.lp = 'left_'
        self.rp = 'right_'
        self.all_cols = [self.lp + col for col in self.cols] + [self.rp + col for col in self.cols]
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device


class WordsPairsProcessor(EMFeatures):
    def __init__(self, df):
        super().__init__(df)
        self.feature_extractor = FeatureExtractor()

    def extract_features(self, model, word_pairs, emb_pairs, train_data_loader):
        model.eval()
        model.to(self.device)
        data_loader = train_data_loader
        data_loader.__init__(word_pairs, emb_pairs)
        word_pair_corrected = data_loader.word_pairs_corrected
        word_pair_corrected['pred'] = model(data_loader.X.to(self.device)).cpu().detach().numpy()
        features = self.feature_extractor.extract_features(word_pair_corrected)
        return features, word_pair_corrected