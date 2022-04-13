import copy
import gc
import os
import pickle
import string
import sys
from functools import partial
from warnings import simplefilter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import display
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier
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

from Evaluation import evaluate_df, correlation_vs_landmark, token_remotion_delta_performance
from FeatureContribution import FeatureContribution
from FeatureExtractor import FeatureExtractor
from Finetune import finetune_BERT
from Modelling import feature_importance
from Net import DatasetAccoppiate, NetAccoppiate, train_model
from WordEmbedding import WordEmbedding
from WordPairGenerator import WordPairGenerator, WordPairGeneratorEdit


class Routine:
    def __init__(self, dataset_name, dataset_path, project_path,
                 reset_files=False, model_name='BERT', device=None, reset_networks=False, clean_special_char=True,
                 col_to_drop=[], model_files_path=None,
                 softlab_path='./content/drive/Shareddrives/SoftLab/',
                 verbose=True, we_finetuned=False,
                 we_finetune_path=None, num_epochs=10,
                 sentence_embedding=True, we=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        simplefilter(action='ignore', category=FutureWarning)
        simplefilter(action='ignore')
        pd.options.display.float_format = '{:.4f}'.format
        pd.options.display.max_rows = 150
        pd.options.display.max_columns = 150
        pd.options.display.max_colwidth = 100
        pd.options.display.precision = 15
        pd.options.display.max_info_columns = 150
        plt.rcParams["figure.figsize"] = (18, 6)
        self.softlab_path = os.path.join(softlab_path)
        self.reset_files = reset_files  # @ param {type:"boolean"}
        self.reset_networks = reset_networks  # @ param {type:"boolean"}
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.feature_extractor = FeatureExtractor()
        self.verbose = verbose
        self.sentence_embedding_dict = None
        self.word_pair_model = None
        self.word_pairing_kind = None

        self.cos_sim = None
        self.feature_kind = None
        if dataset_path is None:
            self.dataset_path = os.path.join(softlab_path, 'Dataset', 'Entity Matching', dataset_name)
        else:
            self.dataset_path = dataset_path
        self.project_path = os.path.join(softlab_path, 'Projects', 'Concept level EM (exclusive-inclluse words)')
        if model_files_path is None:
            self.model_files_path = os.path.join(self.project_path, 'dataset_files', dataset_name, model_name)
        else:
            self.model_files_path = model_files_path
        try:
            os.makedirs(self.model_files_path)
        except Exception as e:
            print(e)
            pass
        try:
            os.makedirs(os.path.join(self.model_files_path, 'results'))
        except Exception as e:
            print(e)
            pass
        self.experiments = {}
        sys.path.append(os.path.join(project_path, 'common_functions'))
        sys.path.append(os.path.join(project_path, 'src'))
        pd.options.display.max_colwidth = 130
        self.train = pd.read_csv(os.path.join(dataset_path, 'train_merged.csv'))
        self.test = pd.read_csv(os.path.join(dataset_path, 'test_merged.csv'))
        self.valid = pd.read_csv(os.path.join(dataset_path, 'valid_merged.csv'))
        if not hasattr(self, 'table_A'):
            self.table_A = pd.read_csv(os.path.join(dataset_path, 'tableA.csv')).drop(col_to_drop, 1)
        if not hasattr(self, 'table_B'):
            self.table_B = pd.read_csv(os.path.join(dataset_path, 'tableB.csv')).drop(col_to_drop, 1)


        if dataset_name == 'BeerAdvo-RateBeer':
            table_A = pd.read_csv(os.path.join(dataset_path, 'tableA.csv'))
            table_B = pd.read_csv(os.path.join(dataset_path, 'tableB.csv'))
            for col in table_A.columns:
                if table_A.dtypes[col] == object:
                    for c, to_replace in [(' _ ™ ', '\''), ('äº _', '\xE4\xBA\xAC'),
                                          ('_ œ', ''), (' _', ''),
                                          ('Ã ', 'Ã'), ('»', ''), ('$', '¤'), (' _ ™ ', '\''),
                                          ('1/2', '½'), ('1/4', '¼'), ('3/4', '¾'),
                                          ('Ãa', '\xC3\xADa'), ('Ä `` ', '\xC4\xAB'), (r'Ã$', '\xC3\xADa'),
                                          ('\. \.', '\.'),
                                          ('Å ¡', '\xC5\xA1'),
                                          ('Ã\'\' ', '\xC3\xBB'), ('_', '\xC3\x80'), ('Ã œ', ''),
                                          ('Ã ¶ ', '\xC3\xB6'), ('Ã»¶ ', '\xC3\xB6'),
                                          (' \.', ''), ('&#40; ', ''), (' \&\#41; ', ''),
                                          ]:
                        table_A[col] = table_A[col].str.replace(c, to_replace)
                        table_B[col] = table_B[col].str.replace(c, to_replace)
                    pat = r"(?P<one>Ã)(?P<two>. )"
                    repl = lambda m: f'Ã{m.group("two")[0]}'
                    table_A[col] = table_A[col].str.replace(pat, repl)
            table_A['Brew_Factory_Name'] = table_A['Brew_Factory_Name'].str.encode('latin-1').str.decode('utf-8')
            self.table_A = table_A
            self.table_B = table_B


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
        self.lp = 'left_'
        self.rp = 'right_'

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
                for char in ['-', '/', '\\']:
                    self.table_A[col] = self.table_A[col].str.replace(char, ' ')
                    self.table_B[col] = self.table_B[col].str.replace(char, ' ')
                self.table_A[col] = self.table_A[col].str.split().str.join(" ").str.lower()
                self.table_B[col] = self.table_B[col].str.split().str.join(" ").str.lower()

        self.table_A = self.table_A.replace('^None$', np.nan, regex=True).replace('^nan$', np.nan, regex=True)
        self.table_B = self.table_B.replace('^None$', np.nan, regex=True).replace('^nan$', np.nan, regex=True)


        self.words_divided = {}
        tmp_path = os.path.join(self.model_files_path, 'words_maps.pickle')
        try:
            assert self.reset_files == False, 'Reset_files'
            with open(tmp_path, 'rb') as file:
                self.words_divided = pickle.load(file)
            print('Loaded ' + tmp_path)
        except Exception as e:
            print(e)
            for name, df in zip(['table_A', 'table_B'], [self.table_A, self.table_B]):
                self.words_divided[name] = WordPairGenerator.map_word_to_attr(df, self.cols, verbose=self.verbose)
            with open(tmp_path, 'wb') as file:
                pickle.dump(self.words_divided, file)

        tmp_cols = ['id', 'left_id', 'right_id', 'label']
        self.train_merged = pd.merge(
            pd.merge(self.train[tmp_cols], self.table_A.add_prefix('left_'), on='left_id'),
            self.table_B.add_prefix('right_'), on='right_id').sort_values('id').reset_index(drop='True')
        self.test_merged = pd.merge(
            pd.merge(self.test[tmp_cols], self.table_A.add_prefix('left_'), on='left_id'),
            self.table_B.add_prefix('right_'), on='right_id').sort_values('id').reset_index(drop='True')
        self.valid_merged = pd.merge(
            pd.merge(self.valid[tmp_cols], self.table_A.add_prefix('left_'), on='left_id'),
            self.table_B.add_prefix('right_'), on='right_id').sort_values('id').reset_index(drop='True')
        for col, type in zip(['id', 'label'], ['UInt32', 'UInt8']):
            self.train_merged[col] = self.train_merged[col].astype(type)
            self.valid_merged[col] = self.valid_merged[col].astype(type)
            self.test_merged[col] = self.test_merged[col].astype(type)

        self.train = self.train_merged
        self.valid = self.valid_merged
        self.test = self.test_merged
        self.sentence_embedding = sentence_embedding
        if we is not None:
            self.we = we
        elif we_finetuned:
            if we_finetune_path is not None:
                finetuned_path = we_finetune_path
            elif we_finetuned == 'SBERT':
                model_save_path = os.path.join(self.model_files_path, 'sBERT')
                if reset_files or os.path.isdir(model_save_path) is False:
                    finetuned_path = finetune_BERT(self, num_epochs=num_epochs, model_save_path=model_save_path)
                else:
                    finetuned_path = model_save_path
            else:
                finetuned_path = os.path.join(self.project_path, 'dataset_files', 'finetuned_models', dataset_name)

            self.we = WordEmbedding(device=self.device, verbose=verbose, model_path=finetuned_path,
                                    sentence_embedding=sentence_embedding)
        else:
            self.we = WordEmbedding(device=self.device, verbose=verbose, sentence_embedding=sentence_embedding)

    # def __del__(self):
    #     try:
    #         for value, key in self.embeddings.items():
    #             value.cpu()
    #     except Exception as e:
    #           print(e)
    #         pass
    #     try:
    #         self.we.mode.to('cpu')
    #     except Exception as e:
    #           print(e)
    #         pass
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     return super(Routine, self).__del__()

    def generate_df_embedding(self, chunk_size=100):
        self.embeddings = {}
        if self.sentence_embedding:
            self.sentence_embedding_dict = {}
        self.words = {}
        try:
            assert self.reset_files == False, 'Reset_files'
            for df_name in ['table_A', 'table_B']:
                tmp_path = os.path.join(self.model_files_path, 'emb_' + df_name + '.csv')
                with open(tmp_path, 'rb') as file:
                    self.embeddings[df_name] = torch.load(file, map_location=torch.device(self.device))
                tmp_path = os.path.join(self.model_files_path, 'words_list_' + df_name + '.csv')
                with open(tmp_path, 'rb') as file:
                    self.words[df_name] = pickle.load(file)
                if self.sentence_embedding:
                    tmp_path = os.path.join(self.model_files_path, 'sentence_emb_' + df_name + '.csv')
                    with open(tmp_path, 'rb') as file:
                        self.sentence_embedding_dict[df_name] = torch.load(file, map_location=torch.device(self.device))
            print('Loaded embeddings.')
        except Exception as e:
            print(e)
            self.we.verbose = self.verbose
            we = self.we
            for name, df in [('table_A', self.table_A), ('table_B', self.table_B)]:
                gc.collect()
                torch.cuda.empty_cache()
                if self.sentence_embedding:
                    emb, words, sentence_emb = we.generate_embedding(df, chunk_size=chunk_size)
                    self.sentence_embedding_dict[name] = sentence_emb
                    tmp_path = os.path.join(self.model_files_path, 'sentence_emb_' + name + '.csv')
                    with open(tmp_path, 'wb') as file:
                        torch.save(sentence_emb, file)
                else:
                    emb, words = we.generate_embedding(df, chunk_size=chunk_size)
                self.embeddings[name] = emb
                self.words[name] = words
                tmp_path = os.path.join(self.model_files_path, 'emb_' + name + '.csv')
                with open(tmp_path, 'wb') as file:
                    torch.save(emb, file)
                tmp_path = os.path.join(self.model_files_path, 'words_list_' + name + '.csv')
                with open(tmp_path, 'wb') as file:
                    pickle.dump(words, file)
        if self.sentence_embedding:
            assert self.sentence_embedding_dict['table_A'][0].shape == torch.Size(
                [
                    768]), f'Sentence emb has shape: {self.sentence_embedding_dict["table_A"][0].shape}. It must be [768]!'

    def get_processed_data(self, df, chunk_size=500, verbose=False):
        we = self.we
        res = {}
        for side in ['left', 'right']:
            gc.collect()
            torch.cuda.empty_cache()
            if verbose:
                print(f'Embedding {side} side')
            prefix = self.lp if side == 'left' else self.rp
            cols = [prefix + col for col in self.cols]
            tmp_df = df.loc[:, cols]
            res[side + '_word_map'] = WordPairGenerator.map_word_to_attr(tmp_df, self.cols, prefix=prefix,
                                                                         verbose=self.verbose)
            if self.sentence_embedding:
                emb, words, sentence_emb = we.generate_embedding(tmp_df, chunk_size=chunk_size)
                res[side + '_sentence_emb'] = sentence_emb
            else:
                emb, words = we.generate_embedding(tmp_df, chunk_size=chunk_size)
            res[side + '_emb'] = emb
            res[side + '_words'] = words
        return res

    def compute_word_pair(self, use_schema=True, **kwargs):
        word_sim = self.word_pairing_kind == 'word_similarity'
        words_pairs_dict, emb_pairs_dict = {}, {}
        if self.sentence_embedding:
            self.sentence_emb_pairs_dict = {}
        try:
            assert self.reset_files == False, 'Reset_files'
            for df_name in ['train', 'valid', 'test']:
                tmp_path = os.path.join(self.model_files_path, df_name + 'word_pairs.csv')
                words_pairs_dict[df_name] = pd.read_csv(tmp_path, keep_default_na=False)

                tmp_path = os.path.join(self.model_files_path, df_name + 'emb_pairs.csv')
                with open(tmp_path, 'rb') as file:
                    emb_pairs_dict[df_name] = pickle.load(file)

                if self.sentence_embedding:
                    tmp_path = os.path.join(self.model_files_path, df_name + 'sentence_emb_pairs.csv')
                    with open(tmp_path, 'rb') as file:
                        self.sentence_emb_pairs_dict[df_name] = pickle.load(file)
            print('Loaded word pairs')
        except Exception as e:
            print(e)

            if word_sim:
                word_pair_generator = WordPairGeneratorEdit(df=self.test, use_schema=use_schema, device=self.device,
                                                            verbose=self.verbose,
                                                            words_divided=self.words_divided,
                                                            sentence_embedding_dict=self.sentence_embedding_dict,
                                                            **kwargs)
            else:
                word_pair_generator = WordPairGenerator(self.words, self.embeddings, words_divided=self.words_divided,
                                                        df=self.test,
                                                        use_schema=use_schema, device=self.device, verbose=self.verbose,
                                                        sentence_embedding_dict=self.sentence_embedding_dict,
                                                        **kwargs)
            for df_name, df in zip(['train', 'valid', 'test'], [self.train, self.valid, self.test]):
                if word_sim:
                    word_pairs = word_pair_generator.process_df(df)
                else:
                    if self.sentence_embedding:
                        word_pairs, emb_pairs, sentence_emb_pairs = word_pair_generator.process_df(df)
                        self.sentence_emb_pairs_dict[df_name] = sentence_emb_pairs
                    else:
                        word_pairs, emb_pairs = word_pair_generator.process_df(df)
                    emb_pairs_dict[df_name] = emb_pairs
                    tmp_path = os.path.join(self.model_files_path, df_name + 'word_pairs.csv')
                    pd.DataFrame(word_pairs).to_csv(tmp_path, index=False)
                    tmp_path = os.path.join(self.model_files_path, df_name + 'emb_pairs.csv')
                    with open(tmp_path, 'wb') as file:
                        pickle.dump(emb_pairs, file)
                    if self.sentence_embedding:
                        tmp_path = os.path.join(self.model_files_path, df_name + 'sentence_emb_pairs.csv')
                        with open(tmp_path, 'wb') as file:
                            pickle.dump(self.sentence_emb_pairs_dict, file)

                words_pairs_dict[df_name] = pd.DataFrame(word_pairs)

        self.words_pairs_dict = words_pairs_dict
        if not word_sim:
            self.emb_pairs_dict = emb_pairs_dict
        if self.sentence_embedding:
            return self.words_pairs_dict, self.emb_pairs_dict, self.sentence_emb_pairs_dict
        else:
            return words_pairs_dict, emb_pairs_dict

    def get_word_pairs(self, df, data_dict, use_schema=True,
                       # parallel=False,
                       **kwargs):
        if self.word_pairing_kind is not None and self.word_pairing_kind == 'word_similarity':
            wp = WordPairGeneratorEdit(df=df, use_schema=use_schema, device=self.device, verbose=self.verbose,
                                       sentence_embedding_dict=self.sentence_embedding_dict, **kwargs)
            res = wp.get_word_pairs(df, data_dict)
            return res
        else:
            wp = WordPairGenerator(df=df, use_schema=use_schema, device=self.device, verbose=self.verbose,
                                   sentence_embedding_dict=self.sentence_embedding_dict, **kwargs)
        # if parallel:
        #     res = wp.get_word_pairs_parallel(df, data_dict)
        # else:
        res = wp.get_word_pairs(df, data_dict)  # empty lost
        if self.sentence_embedding:
            word_pairs, emb_pairs, sent_emb_pairs = res
        else:
            word_pairs, emb_pairs = res
        word_pairs = pd.DataFrame(word_pairs)
        if self.sentence_embedding:
            return word_pairs, emb_pairs, sent_emb_pairs
        else:
            return word_pairs, emb_pairs

    def net_train(self, num_epochs=40, lr=3e-5, batch_size=256, word_pairs=None, emb_pairs=None,
                  sentence_emb_pairs=None,
                  valid_pairs=None, valid_emb=None, valid_sentence_emb_pairs=None):
        if word_pairs is None or emb_pairs is None:
            word_pairs = self.words_pairs_dict['train']
            emb_pairs = self.emb_pairs_dict['train']
            if self.sentence_embedding:
                sentence_emb_pairs = self.sentence_emb_pairs_dict['train']
            else:
                sentence_emb_pairs = None

        if valid_pairs is None or valid_emb is None:
            valid_pairs = self.words_pairs_dict['valid']
            valid_emb = self.emb_pairs_dict['valid']
            if self.sentence_embedding:
                valid_sententce_emb_pairs = self.sentence_emb_pairs_dict['valid']
            else:
                valid_sententce_emb_pairs = None
        data_loader = DatasetAccoppiate(word_pairs, emb_pairs, sentence_embedding_pairs=sentence_emb_pairs)
        self.train_data_loader = data_loader
        best_model = NetAccoppiate(sentence_embedding=self.sentence_embedding, )
        device = self.device
        tmp_path = os.path.join(self.model_files_path, 'net0.pickle')
        try:
            assert self.reset_networks == False, 'resetting networks'
            best_model.load_state_dict(torch.load(tmp_path, map_location=torch.device(device)))
        except Exception as e:
            print(e)
            net = NetAccoppiate(sentence_embedding=self.sentence_embedding)
            net.to(device)
            criterion = nn.BCELoss().to(device)
            # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=.9)
            optimizer = optim.Adam(net.parameters(), lr=lr)

            train_dataset = data_loader
            valid_dataset = copy.deepcopy(train_dataset)
            valid_dataset.__init__(valid_pairs, valid_emb, sentence_embedding_pairs=valid_sententce_emb_pairs)

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

    def preprocess_word_pairs(self, **kwargs):
        features_dict = {}
        words_pairs_dict = {}
        for name in ['train', 'valid', 'test']:
            if self.sentence_embedding:
                sentence_emb_pairs = self.sentence_emb_pairs_dict[name]
            else:
                sentence_emb_pairs = None
            if self.word_pairing_kind == 'word_similarity':
                feat, word_pairs = self.extract_features(None, self.words_pairs_dict[name],
                                                         None, None,
                                                         sentence_emb_pairs=sentence_emb_pairs, **kwargs)
            else:
                feat, word_pairs = self.extract_features(self.word_pair_model, self.words_pairs_dict[name],
                                                         self.emb_pairs_dict[name], self.train_data_loader,
                                                         sentence_emb_pairs=sentence_emb_pairs, **kwargs)
            features_dict[name] = feat
            words_pairs_dict[name] = word_pairs
        self.features_dict, self.words_pairs_dict = features_dict, words_pairs_dict
        return features_dict, words_pairs_dict

    def extract_features(self, model: NetAccoppiate, word_pairs, emb_pairs, train_data_loader, sentence_emb_pairs=None,
                         **kwargs):
        if self.cos_sim is not None or self.word_pairing_kind == 'word_similarity':
            if self.cos_sim == 'binary':
                word_pair_corrected = word_pairs
                word_pair_corrected['pred'] = np.where(word_pair_corrected['cos_sim'] > 0, 1, 0)
            else:
                word_pair_corrected = word_pairs
                word_pair_corrected['pred'] = word_pair_corrected['cos_sim']
        else:
            model.eval()
            model.to(self.device)
            data_loader = train_data_loader
            data_loader.__init__(word_pairs, emb_pairs, sentence_emb_pairs)
            word_pair_corrected = data_loader.word_pairs_corrected
            with torch.no_grad():
                word_pair_corrected['pred'] = model(data_loader.X.to(self.device)).cpu().detach().numpy()

            # features = self.feature_extractor.extract_features(word_pair_corrected, **kwargs)
        if self.feature_kind == 'min':
            features = self.feature_extractor.extract_features_min(word_pair_corrected, **kwargs)
        else:
            features = self.feature_extractor.extract_features_by_attr(word_pair_corrected, self.cols, **kwargs)
        return features, word_pair_corrected

    def EM_modelling(self, *args, do_evaluation=False, do_feature_selection=False, results_path='results'):

        if hasattr(self, 'models') == False:
            mmScaler = MinMaxScaler()
            mmScaler.clip = False
            self.models = [
                ('LR',
                 Pipeline([('mm', copy.copy(mmScaler)), ('LR', LogisticRegression(max_iter=200, random_state=0))])),
                ('LDA', Pipeline([('mm', copy.copy(mmScaler)), ('LDA', LinearDiscriminantAnalysis())])),
                ('KNN', Pipeline([('mm', copy.copy(mmScaler)), ('KNN', KNeighborsClassifier())])),
                ('CART', DecisionTreeClassifier(random_state=0)),
                ('NB', GaussianNB()),
                ('SVM', Pipeline([('mm', copy.copy(mmScaler)), ('SVM', SVC(probability=True, random_state=0))])),
                ('AB', AdaBoostClassifier(random_state=0)),
                ('GBM', GradientBoostingClassifier(random_state=0)),
                ('RF', RandomForestClassifier(random_state=0)),
                ('ET', ExtraTreesClassifier(random_state=0)),
                ('dummy', DummyClassifier(strategy='stratified', random_state=0)),
            ]
        # models.append(('Vote', VotingClassifier(models[:-1], voting='soft')))
        model_names = [x[0] for x in self.models]

        X_train, y_train = self.features_dict['train'].to_numpy(), self.train.label.astype(int)
        X_valid, y_valid = self.features_dict['valid'].to_numpy(), self.valid.label.astype(int)
        X_test, y_test = self.features_dict['test'].to_numpy(), self.test.label.astype(int)

        res = {(x, y): [] for x in ['train', 'valid', 'test'] for y in ['f1', 'precision', 'recall']}
        df_pred = {}
        for name, model in tqdm(self.models):
            model.fit(X_train, y_train)
            for turn_name, turn_df, turn_y in zip(['train', 'valid', 'test'], [X_train, X_valid, X_test],
                                                  [y_train, y_valid, y_test]):
                df_pred[turn_name] = model.predict(turn_df)
                for score_name, scorer in [['f1', f1_score], ['precision', precision_score], ['recall', recall_score]]:
                    res[(turn_name, score_name)].append(scorer(turn_y, df_pred[turn_name]))

        print('before feature selection')
        res_df = pd.DataFrame(res, index=model_names)
        res_df.index.name = 'model_name'
        try:
            os.makedirs(os.path.join(self.model_files_path, results_path))
        except Exception as e:
            print(e)
            pass
        res_df.to_csv(os.path.join(self.model_files_path, results_path, 'performances.csv'))
        display(res_df)
        best_f1 = res_df[('test', 'f1')].max()
        best_features = self.features_dict['train'].columns
        best_model_name = res_df.iloc[[res_df[('test', 'f1')].argmax()]].index.values[0]
        for x in self.models:
            if x[0] == best_model_name:
                best_model = x[1]

        # Feature selection
        if do_feature_selection:
            print('running feature score')
            score_df = {'feature': [], 'score': []}
            X_train, y_train = self.features_dict['train'], self.train.label.astype(int)
            X_valid, y_valid = self.features_dict['valid'], self.valid.label.astype(int)
            X_test, y_test = self.features_dict['test'], self.test.label.astype(int)

            cols = self.features_dict['train'].columns
            new_cols = cols
            different = True
            iter = 0
            while different and iter <= 2:
                cols = new_cols
                score_df, res_df, new_cols = feature_importance(X_train, y_train, X_valid, y_valid, cols)
                different = len(cols) != len(new_cols)
                iter += 1

            self.score_df = score_df
            self.res_df = res_df
            selected_features = new_cols

            res = {(x, y): [] for x in ['train', 'valid', 'test'] for y in ['f1', 'precision', 'recall']}
            print('Running models')
            for name, model in tqdm(self.models):
                model.fit(X_train, y_train)
                for turn_name, turn_df, turn_y in zip(['train', 'valid', 'test'], [X_train, X_valid, X_test],
                                                      [y_train, y_valid, y_test]):
                    df_pred[turn_name] = model.predict(turn_df)
                    for score_name, scorer in [['f1', f1_score], ['precision', precision_score],
                                               ['recall', recall_score]]:
                        res[(turn_name, score_name)].append(scorer(turn_y, df_pred[turn_name]))
            self.models = self.models
            res_df = pd.DataFrame(res, index=model_names)
            res_df.index.name = 'model_name'
            display(res_df)

            if best_f1 < res_df[('test', 'f1')].max():
                best_f1 = res_df[('test', 'f1')].max()
                best_features = selected_features
                best_model_name = res_df.iloc[[res_df[('test', 'f1')].argmax()]].index.values[0]
                for x in self.models:
                    if x[0] == best_model_name:
                        best_model = x[1]

                res_df.to_csv(os.path.join(self.model_files_path, results_path, 'performances.csv'))

        X_train, y_train = self.features_dict['train'][best_features].to_numpy(), self.train.label.astype(int)
        best_model.fit(X_train, y_train)
        model_data = {'features': best_features, 'model': best_model}
        tmp_path = os.path.join(self.model_files_path, 'best_feature_model_data.pickle')
        self.best_model_data = model_data
        with open(tmp_path, 'wb') as file:
            pickle.dump(model_data, file)

        linear_model = Pipeline([('LR', LogisticRegression(max_iter=200, random_state=0))])
        # LogisticRegression(max_iter=200, random_state=0)
        X_train, y_train = self.features_dict['train'][best_features].to_numpy(), self.train.label.astype(int)
        linear_model.fit(X_train, y_train)
        model_data = {'features': best_features, 'model': linear_model}
        tmp_path = os.path.join(self.model_files_path, 'linear_model.pickle')
        with open(tmp_path, 'wb') as file:
            pickle.dump(model_data, file)
        # save unit_contribution
        # co = linear_model['LR'].coef_
        # co_df = pd.DataFrame(co, columns=self.features_dict['valid'].columns).T
        # for name in ['train','valid']
        # turn_contrib = FeatureContribution.extract_features_by_attr(word_relevance, routine.cols)
        # for x in co_df.index:
        #     turn_contrib[x] = turn_contrib[x] * co_df.loc[x, 0]
        # data = turn_contrib.sum(1).to_numpy().reshape(-1, 1)
        # word_relevance['token_contribution'] = data

        if do_evaluation:
            self.evaluation(self.valid_merged)
        return res_df

    def get_match_score(self, features_df, lr=False, reload=False):

        if lr is True:
            tmp_path = os.path.join(self.model_files_path, 'linear_model.pickle')
        else:
            tmp_path = os.path.join(self.model_files_path, 'best_feature_model_data.pickle')
        if not hasattr(self, 'best_model_data') or reload:
            with open(tmp_path, 'rb') as file:
                model_data = pickle.load(file)
            self.best_model_data = model_data
        self.model = self.best_model_data['model']
        X = features_df[self.best_model_data['features']].to_numpy()
        if isinstance(self.model, Pipeline) and isinstance(self.model[0], MinMaxScaler):
            self.model[0].clip = False
        return self.model.predict_proba(X)[:, 1]

    def plot_rf(self, rf, columns):
        pd.DataFrame([rf.feature_importances_], columns=columns).T.plot.bar(figsize=(25, 5));

    def get_relevance_scores(self, word_pairs, emb_pairs, sentence_emb_pairs=None, **kwargs):  # m2

        feat, word_pairs = self.extract_features(emb_pairs=emb_pairs, word_pairs=word_pairs,
                                                 model=self.word_pair_model, train_data_loader=self.train_data_loader,
                                                 sentence_emb_pairs=sentence_emb_pairs,
                                                 **kwargs)

        return feat, word_pairs

    def get_predictor(self):
        self.reset_networks = False
        self.net_train()

        def predictor(df_to_process, routine, return_data=False, lr=False, chunk_size=500, reload=False):

            df_to_process = df_to_process.copy()
            if 'id' not in df_to_process.columns:
                df_to_process = df_to_process.reset_index(drop=True)
                df_to_process['id'] = df_to_process.index
            gc.collect()
            torch.cuda.empty_cache()
            data_dict = routine.get_processed_data(df_to_process, chunk_size=chunk_size)
            res = routine.get_word_pairs(df_to_process, data_dict)
            features, word_relevance = routine.get_relevance_scores(*res)

            if lr:
                match_score = routine.get_match_score(features, lr=lr, reload=reload)
            else:
                match_score = routine.get_match_score(features, reload=reload)
            match_score_series = pd.Series(0.5, index=df_to_process.id)
            match_score_series[features.index] = match_score
            match_score = match_score_series.values

            if return_data:
                if lr:
                    lr = routine.model['LR']
                    co = lr.coef_
                    co_df = pd.DataFrame(co, columns=routine.features_dict['test'].columns).T
                    turn_contrib = FeatureContribution.extract_features_by_attr(word_relevance, routine.cols)
                    for x in co_df.index:
                        turn_contrib[x] = turn_contrib[x] * co_df.loc[x, 0]
                    data = turn_contrib.sum(1).to_numpy().reshape(-1, 1)
                    word_relevance['token_contribution'] = data
                return match_score, data_dict, res, features, word_relevance
            else:
                return match_score

        return partial(predictor, routine=self)

    def get_calculated_data(self, df_name):
        features = self.features_dict[df_name]
        word_relevance = self.words_pairs_dict[df_name]
        pred = self.get_match_score(features, lr=True, reload=True)
        lr = self.model['LR']
        co = lr.coef_
        co_df = pd.DataFrame(co, columns=self.features_dict[df_name].columns).T
        turn_contrib = FeatureContribution.extract_features_by_attr(word_relevance, self.cols)
        for x in co_df.index:
            turn_contrib[x] = turn_contrib[x] * co_df.loc[x, 0]
        data = turn_contrib.sum(1).to_numpy().reshape(-1, 1)
        word_relevance['token_contribution'] = data
        return pred, features, word_relevance

    def evaluation(self, df: pd.DataFrame, pred_threshold=0.00, plot=True, operations=[0, 1, 2, 3], score_col='pred',
                   chunk_size=400, reset=False):
        self.reset_networks = False
        self.net_train()

        predictor = self.get_predictor()
        predictor = partial(predictor, chunk_size=chunk_size, lr=True)
        # pred = predictor(df)
        # tmp_df = df[(pred > 0.5)]
        # max_len = min(100, tmp_df.shape[0])
        # df_to_process = tmp_df.sample(max_len, random_state=0).replace(pd.NA, '').reset_index(drop=True)
        # df_to_process['id'] = df_to_process.index
        # self.ev_df['match'] = df_to_process
        df = df.copy().replace(pd.NA, '')
        if df.equals(self.valid_merged.replace(pd.NA, '')):
            print('Evaluating valid dataset')
            df_name = 'valid'
            pred, features, word_relevance = self.get_calculated_data(df_name)
        else:
            pred, data_dict, res, features, word_relevance = predictor(df, return_data=True, lr=True,
                                                                       chunk_size=chunk_size, reload=True)

        self.ev_df = {}

        match_df = df[df['label'] > .5]
        sample_len = min(100, match_df.shape[0])
        match_ids = match_df.id.sample(sample_len).values
        self.ev_df['match'] = match_df[match_df.id.isin(match_ids)]

        no_match_df = df[(df['label'] < 0.5) & (pred >= pred_threshold)]
        sample_len = min(100, no_match_df.shape[0])
        no_match_ids = no_match_df.id.sample(sample_len).values
        self.ev_df['nomatch'] = no_match_df[no_match_df.id.isin(no_match_ids)]
        if 0 in operations:
            # token_remotion_delta_performance
            tmp_path = os.path.join(self.model_files_path, 'results',
                                    f'{score_col}__evaluation_token_remotion_delta_performance.csv')
            if reset is False and os.path.isfile(tmp_path):
                delta_performance = pd.read_csv(tmp_path)
            else:
                delta_performance = token_remotion_delta_performance(df, df.label.values.astype(int), word_relevance,
                                                                     predictor, plot=plot, score_col=score_col)
                delta_performance.to_csv(
                    tmp_path)
            display(delta_performance)
            self.delta_performance = delta_performance

        if 1 in operations:
            self.verbose = False
            self.we.verbose = False
            if reset is False and os.path.isfile(
                    os.path.join(self.model_files_path, 'results', f'{score_col}__evaluation_no_match_mean_delta.csv')):
                pass
            else:
                # Evaluate impacts with words remotion
                res_df = evaluate_df(word_relevance[word_relevance.id.isin(match_ids)],
                                     match_df[match_df.id.isin(match_ids)],
                                     predictor, score_col=score_col)

                res_df['concorde'] = (res_df['detected_delta'] > 0) == (res_df['expected_delta'] > 0)
                self.experiments['exp1_match'] = res_df
                match_stat = res_df.groupby('comb_name')[['concorde']].mean()
                match_stat.to_csv(os.path.join(self.model_files_path, 'results', f'{score_col}__evaluation_match.csv'))
                display(match_stat)
                res_df_match = res_df
                match_stat = res_df.groupby('comb_name')[['detected_delta']].agg(
                    ['size', 'mean', 'median', 'min', 'max'])
                match_stat.to_csv(
                    os.path.join(self.model_files_path, 'results', f'{score_col}__evaluation_match_mean_delta.csv'))

                res_df = evaluate_df(word_relevance[word_relevance.id.isin(no_match_ids)],
                                     no_match_df[no_match_df.id.isin(no_match_ids)],
                                     predictor, score_col=score_col)
                res_df['concorde'] = (res_df['detected_delta'] > 0) == (res_df['expected_delta'] > 0)
                self.experiments['exp1_no_match'] = res_df
                no_match_stat = res_df.groupby('comb_name')[['concorde']].mean()
                no_match_stat.to_csv(
                    os.path.join(self.model_files_path, 'results', f'{score_col}__evaluation_no_match.csv'))
                display(no_match_stat)
                res_df_no_match = res_df
                no_match_stat = res_df.groupby('comb_name')[['detected_delta']].agg(
                    ['size', 'mean', 'median', 'min', 'max'])
                no_match_stat.to_csv(
                    os.path.join(self.model_files_path, 'results', f'{score_col}__evaluation_no_match_mean_delta.csv'))

                res_df_match.to_csv(
                    os.path.join(self.model_files_path, 'results', f'{score_col}__evaluation_match_combinations.csv'))
                res_df_no_match.to_csv(
                    os.path.join(self.model_files_path, 'results',
                                 f'{score_col}__evaluation_no_match_combinations.csv'))
            self.verbose = True
            self.we.verbose = True
        if 2 in operations:
            self.verbose = False
            self.we.verbose = False
            tmp_path = os.path.join(self.model_files_path, 'results',
                                    f'{score_col}__evaluation_correlation_vs_landmark.csv')
            if reset is False and os.path.isfile(tmp_path):
                correlation_data = pd.read_csv(tmp_path)
            else:

                # Correlation between relevance and landmark impacts
                correlation_data = correlation_vs_landmark(df, word_relevance, predictor, match_ids, no_match_ids,
                                                           score_col=score_col)
                correlation_data.to_csv(tmp_path)
            display(correlation_data)
            self.correlation_data = correlation_data
            self.verbose = True
            self.we.verbose = True
        if 3 in operations:
            def cumsum_perc(df, n_perc_units=[1, 5, 10, 20, 30, 40, 50]):
                x = df['token_contribution'].abs()
                tot_sum = x.sum()
                n_units = x.count()
                cum_sum = x.sort_values(ascending=False).cumsum()
                d = {}
                for k in n_perc_units:
                    d[int(k)] = cum_sum.iloc[np.ceil(k / 100 * n_units).astype(int) - 1] / tot_sum
                return pd.Series(d, index=d.keys())

            n_perc_units = np.concatenate([[1, 3, 5], np.linspace(10, 100, 10)])
            tmp = word_relevance.groupby(
                ['id', np.where(word_relevance['token_contribution'] > 0, 'Positive', 'Negative')]).apply(
                partial(cumsum_perc, n_perc_units=n_perc_units))
            to_plot = tmp.groupby(tmp.index.get_level_values(1)).mean()
            to_plot.loc[:, 0] = 0
            to_plot = to_plot.T.sort_index()
            self.coinciveness = to_plot

            tmp = word_relevance.groupby('id').apply(partial(cumsum_perc, n_perc_units=n_perc_units))
            to_plot = tmp.mean()
            to_plot.loc[0] = 0
            to_plot = to_plot.T.sort_index()
            self.coinciveness['total'] = to_plot
            to_plot = self.coinciveness
            tmp_path = os.path.join(self.model_files_path, 'results', f'{score_col}__pareto-relation.csv')
            self.coinciveness.to_csv(tmp_path)

            import seaborn as sns;
            sns.set()  # for plot styling
            sns.set(rc={"figure.dpi": 100, 'savefig.dpi': 300})
            sns.set_context('notebook')
            sns.set_style('whitegrid')
            to_plot.plot()
            plt.ylim(0, 1)
            plt.xticks(to_plot.index, to_plot.index);
            plt.xlim(to_plot.index[0] / 2, to_plot.index[-1])
            plt.tight_layout()
        if 4 in operations:  # AOPC

            pass

        return word_relevance  # res_df_match, res_df_no_match, delta_performance, correlation_data
