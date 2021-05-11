import copy
import os
import pickle
import time
from multiprocessing import Pool

import torch.nn as nn
import torch.optim as optim

from dataset_loader import *
from feature_extractor import FeatureExtractor, FeatureExtractorOOV
from networks import NetSpaiate, NetAccoppiate, train_model



class RoutineUtility:
    def __init__(self, model_results_path, dataset_path, word_vectors,
                 exclude_attrs=['id', 'left_id', 'right_id', 'label'], reset_files=False, n_proc=1):
        self.word_vectors = word_vectors
        self.reset_files = reset_files
        self.n_proc = n_proc
        self.model_results_path = model_results_path
        self.dataset_path = dataset_path
        self.exclude_attrs = exclude_attrs
        self.preprocess_loader = None
        self.in_vocab_splitted = None
        self.oov_df = None
        self.in_vocab_df = None
        self.cols = None
        self.nets = {}
        self.names = {'best_net_unpaired': 'net_unpaired_best.pickle',
                      'last_net_unpaired': 'net_unpaired_last.pickle',
                      'best_net_paired': 'net_paired_best.pickle',
                      'last_net_paired': 'net_paired_last.pickle'
                      }

    def prepare_datasets(self):
        try:
            assert self.reset_files is False, 'Reset Files'
            pd.read_csv(os.path.join(self.model_results_path, 'tableA_in_vocab.csv'))
            pd.read_csv(os.path.join(self.model_results_path, 'tableB_in_vocab.csv'))
        except Exception as e:
            print(e)
            table_A = pd.read_csv(os.path.join(self.dataset_path, 'tableA.csv'))
            table_B = pd.read_csv(os.path.join(self.dataset_path, 'tableB.csv'))
            start_time = time.time()
            if self.n_proc == 1:
                tmp_res = [self.preprocess_row(x) for x in table_A.iterrows()]
            else:
                pool = Pool(2)
                tmp_res = pool.map(self.preprocess_row, table_A.iterrows())
                pool.close()
                pool.join()
            print(f'--- {time.time() - start_time} seconds ---')
            in_vocab = [x[0] for x in tmp_res]
            non_in_vocab = [x[1] for x in tmp_res]
            pd.DataFrame(in_vocab).to_csv(os.path.join(self.model_results_path, 'tableA_in_vocab.csv'), index=False)
            pd.DataFrame(non_in_vocab).to_csv(os.path.join(self.model_results_path, 'tableA_non_in_vocab.csv'),
                                              index=False)

            start_time = time.time()
            if self.n_proc == 1:
                tmp_res = [self.preprocess_row(x) for x in table_B.iterrows()]
            else:
                pool = Pool(2)
                tmp_res = pool.map(self.preprocess_row, table_B.iterrows())
                pool.close()
                pool.join()
            print(f'--- {time.time() - start_time} seconds ---')
            in_vocab = [x[0] for x in tmp_res]
            non_in_vocab = [x[1] for x in tmp_res]
            pd.DataFrame(in_vocab).to_csv(os.path.join(self.model_results_path, 'tableB_in_vocab.csv'), index=False)
            pd.DataFrame(non_in_vocab).to_csv(os.path.join(self.model_results_path, 'tableB_non_in_vocab.csv'),
                                              index=False)
        A_in_vocab = pd.read_csv(os.path.join(self.model_results_path, 'tableA_in_vocab.csv'))
        B_in_vocab = pd.read_csv(os.path.join(self.model_results_path, 'tableB_in_vocab.csv'))
        self.cols = np.setdiff1d(A_in_vocab.columns, ['id'])
        train = pd.read_csv(os.path.join(self.dataset_path, 'train.csv'))
        test = pd.read_csv(os.path.join(self.dataset_path, 'test.csv'))
        valid = pd.read_csv(os.path.join(self.dataset_path, 'valid.csv'))
        train.columns = ['left_id', 'right_id', 'label']
        test.columns = ['left_id', 'right_id', 'label']
        valid.columns = ['left_id', 'right_id', 'label']
        A_in_vocab = A_in_vocab.add_prefix('left_')
        B_in_vocab = B_in_vocab.add_prefix('right_')
        train_merged = pd.merge(pd.merge(train, A_in_vocab, on='left_id', suffixes=('', '')), B_in_vocab, on='right_id',
                                suffixes=('', ''))
        test_merged = pd.merge(pd.merge(test, A_in_vocab, on='left_id', suffixes=('', '')), B_in_vocab, on='right_id',
                               suffixes=('', ''))
        valid_merged = pd.merge(pd.merge(valid, A_in_vocab, on='left_id', suffixes=('', '')), B_in_vocab, on='right_id',
                                suffixes=('', ''))
        train_merged.to_csv(os.path.join(self.model_results_path, 'train_merged_in_vocab.csv'), index_label='id')
        test_merged.to_csv(os.path.join(self.model_results_path, 'test_merged_in_vocab.csv'), index_label='id')
        valid_merged.to_csv(os.path.join(self.model_results_path, 'valid_merged_in_vocab.csv'), index_label='id')
        self.in_vocab_df = {'train': train_merged, 'test': test_merged, 'valid': valid_merged}

        A_non_in_vocab = pd.read_csv(os.path.join(self.model_results_path, 'tableA_non_in_vocab.csv'))
        B_non_in_vocab = pd.read_csv(os.path.join(self.model_results_path, 'tableB_non_in_vocab.csv'))
        A_non_in_vocab = A_non_in_vocab.add_prefix('left_')
        B_non_in_vocab = B_non_in_vocab.add_prefix('right_')
        train_merged = pd.merge(pd.merge(train, A_non_in_vocab, on='left_id', suffixes=('', '')), B_non_in_vocab,
                                on='right_id', suffixes=('', ''))
        test_merged = pd.merge(pd.merge(test, A_non_in_vocab, on='left_id', suffixes=('', '')), B_non_in_vocab,
                               on='right_id', suffixes=('', ''))
        valid_merged = pd.merge(pd.merge(valid, A_non_in_vocab, on='left_id', suffixes=('', '')), B_non_in_vocab,
                                on='right_id', suffixes=('', ''))
        train_merged.to_csv(os.path.join(self.model_results_path, 'train_merged_non_in_vocab.csv'), index_label='id')
        test_merged.to_csv(os.path.join(self.model_results_path, 'test_merged_non_in_vocab.csv'), index_label='id')
        valid_merged.to_csv(os.path.join(self.model_results_path, 'valid_merged_non_in_vocab.csv'), index_label='id')
        self.oov_df = {'train': train_merged, 'test': test_merged, 'valid': valid_merged}

    def preprocess_row(self, item):
        index, el = item
        el_words, OOV_words = self.divide_words_in_vocab(el)
        el_words['id'] = index
        OOV_words['id'] = index
        return el_words, OOV_words

    def divide_words_in_vocab(self, el):
        el_words = {col: '' for col in self.cols}
        OOV_words = copy.deepcopy(el_words)
        for col in self.cols:
            if el.notnull()[col]:
                for word in str(el[col]).split():
                    if self.word_vectors.__contains__(word):
                        el_words[col] += word + ' '
                    else:
                        OOV_words[col] += word + ' '
                el_words[col] = el_words[col][:-1]
                OOV_words[col] = OOV_words[col][:-1]
        return el_words, OOV_words

    def __routine_split(self):
        return self.routine_split(self.in_vocab_df)

    def routine_split(self, dict_df, names=['train', 'test', 'valid']):
        res = {}
        for name in names:
            tmp_path = os.path.join(self.model_results_path, name + '_tokens.pickle')
            try:
                assert self.reset_files == False, 'Reset_files'
                with open(tmp_path, 'rb') as file:
                    tmp = pickle.load(file)
                print('Loaded ' + name)
            except Exception as e:
                print(e)
                tmp = FeatureExtractor.split_paired_unpaired(dict_df[name], self.word_vectors, n_proc=1)
                with open(tmp_path, 'wb') as file:
                    pickle.dump(tmp, file)
            res[name] = {'paired': tmp[0], 'unpaired': tmp[1]}
        self.in_vocab_splitted = res
        return res


    def __get_loaders(self):
        return self.get_loaders(self.in_vocab_splitted['train']['paired'], self.in_vocab_splitted['train']['unpaired'])

    def get_loaders(self, train_paired, train_unpaired):
        tmp_path_p = os.path.join(self.model_results_path, 'DatasetAccoppiate_Loader.pickle')
        tmp_path_unp = os.path.join(self.model_results_path, 'DatasetSpaiate_Loader.pickle')
        try:
            assert self.reset_files == False, 'Reset_files'
            with open(tmp_path_p, 'rb') as file:
                tmp_p = pickle.load(file)
            with open(tmp_path_unp, 'rb') as file:
                tmp_unp = pickle.load(file)
            print('Loaded')
        except Exception as e:
            print(e)
            tmp_p = DatasetAccoppiate(train_unpaired, self.word_vectors)
            tmp_unp = DatasetSpaiate(train_paired, self.word_vectors)
            with open(tmp_path_p, 'wb') as file:
                pickle.dump(tmp_p, file)
            with open(tmp_path_unp, 'wb') as file:
                pickle.dump(tmp_unp, file)
                print('saved.')
        self.preprocess_loader = {'paired': tmp_p, 'unpaired': tmp_unp}
        return tmp_p, tmp_unp

    def save_net(self, net, name):
        tmp_path = os.path.join(self.model_results_path, name)
        print('Save...')
        torch.save(net.state_dict(), tmp_path)

    def routine_net(self, paired, device, reset_networks=True):
        assert paired in ['paired', 'unpaired'], 'paired must be \'paired\' or \'unpaired\' '
        try:
            assert reset_networks == False, 'resetting networks'
            print('NO retraining.')
            net_class = NetAccoppiate if paired == 'paired' else NetSpaiate
            best_model, last_model = net_class(), net_class()
            best_model.load_state_dict(
                torch.load(os.path.join(self.model_results_path, self.names['best_net_' + paired]),
                           map_location=torch.device(device)))
            last_model.load_state_dict(
                torch.load(os.path.join(self.model_results_path, self.names['last_net_' + paired]),
                           map_location=torch.device(device)))
            print('Loaded')
        except Exception as e:
            if paired == 'paired':
                res = self.train_paired(self.in_vocab_splitted, self.preprocess_loader[paired], device)
            else:
                res = self.train_unpaired(self.in_vocab_splitted, self.preprocess_loader[paired], device)
            best_model, score_history, last_model = res
            self.save_net(best_model, self.names['best_net_' + paired])
            self.save_net(last_model, self.names['last_net_' + paired])
        self.nets[paired + '_best'] = best_model
        self.nets[paired + '_last'] = last_model

    def train_unpaired(self, in_vocab_splitted, train_loader_upaired, device):
        net = NetSpaiate()
        net.to(device)
        # criterion = nn.MSELoss().to(device)
        criterion = nn.BCELoss().to(device)
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=.9)
        # optimizer = optim.Adam(net.parameters(), lr=0.001)
        batch_size = 32

        train_dataset = train_loader_upaired
        valid_dataset = copy.deepcopy(train_dataset)
        valid_dataset.__init__(in_vocab_splitted['valid']['unpaired'], self.word_vectors)
        dataloaders_dict = {'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
                            'valid': DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)}

        best_model, score_history, last_model = train_model(net,
                                                            dataloaders_dict, criterion, optimizer,
                                                            nn.MSELoss().to(device), num_epochs=200, device=device)
        out = net(valid_dataset.X.to(device))
        print(f'best_valid --> mean:{out.mean():.4f}  std: {out.std():.4f}')
        out = last_model(valid_dataset.X.to(device))
        print(f'last_model --> mean:{out.mean():.4f}  std: {out.std():.4f}')

        return best_model, score_history, last_model

    def train_paired(self, in_vocab_splitted, train_loader_paired, device):
        batch_size = 32
        net = NetAccoppiate()
        net.to(device)
        criterion = nn.BCELoss().to(device)
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=.9)

        train_dataset = train_loader_paired
        valid_dataset = copy.deepcopy(train_dataset)
        valid_dataset.__init__(in_vocab_splitted['valid']['paired'], self.word_vectors)
        dataloaders_dict = {'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
                            'valid': DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)}

        best_model, score_history, last_model = train_model(net,
                                                            dataloaders_dict, criterion, optimizer,
                                                            nn.MSELoss().to(device), num_epochs=200, device=device)
        out = net(valid_dataset.X.to(device))
        print(f'best_valid --> mean:{out.mean():.4f}  std: {out.std():.4f}')
        out = last_model(valid_dataset.X.to(device))
        print(f'last_model --> mean:{out.mean():.4f}  std: {out.std():.4f}')

        return best_model, score_history, last_model

    def get_ready_for_feature_extraction(self, device):
        self.prepare_datasets()
        self.__routine_split()
        self.__get_loaders()
        self.routine_net('paired', device, reset_networks=False)
        self.routine_net('unpaired', device, reset_networks=False)

    def routine_extract(self, complementary=True):
        net_paired = self.nets['paired_best']
        loader_paired = self.preprocess_loader['paired']
        net_unpaired = self.nets['unpaired_best']
        loader_unpaired = self.preprocess_loader['unpaired']
        train = self.in_vocab_df['train']
        processor = FeatureExtractor(net_paired, loader_paired, net_unpaired, loader_unpaired, df=train,
                                     word_vectors=self.word_vectors)
        processor_oov = FeatureExtractorOOV(train, self.word_vectors)

        train_paired_unpaired = processor.generate_paired_unpaired(*(self.in_vocab_splitted['train'].values()), device='cpu')
        test_paired_unpaired = processor.generate_paired_unpaired(*(self.in_vocab_splitted['test'].values()), device='cpu')
        train_stat = processor.extract_features(*train_paired_unpaired, train, complementary)
        test_stat = processor.extract_features(*test_paired_unpaired, self.in_vocab_df['test'], complementary)

        train_stat_oov = processor_oov.process(self.oov_df['train'])
        test_stat_oov = processor_oov.process(self.oov_df['test'])

        x, y = train_stat_oov, train_stat
        train_stat_merged = x.merge(y.set_index('id').drop(['label'], axis=1), on='id', suffixes=('_oov', ''))
        x, y = test_stat_oov, test_stat
        test_stat_merged = x.merge(y.set_index('id').drop(['label'], axis=1), on='id', suffixes=('_oov', ''))
        return train_stat_merged, test_stat_merged


