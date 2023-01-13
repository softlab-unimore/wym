from notebook_import_utility_env import *
import pickle
from copy import deepcopy

from tqdm.autonotebook import tqdm

from dataset_names import sorted_dataset_names
import os
import sys
from datetime import datetime

import pandas as pd
from warnings import simplefilter
from wym.BERTRoutine import Routine
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore')



def run_pipeline(dataset_name, routine):

    # pd.DataFrame(routine.time_list_dict) # todo add sbert time to dict

    dataset_data_dict = {}
    time_list_dict = []
    time_dict = {}
    to_iter = [['train', routine.train_merged], ['test', routine.test_merged],
               ['valid', routine.valid_merged]]
    routine.train_merged = to_iter[0][1]
    routine.test_merged = to_iter[1][1]
    routine.valid_merged = to_iter[2][1]

    for name, df_to_process in to_iter:
        df_to_process = df_to_process.copy()
        time_dict.update(**{'df_name': dataset_name, 'df_split': name,
                            'size': df_to_process.shape[0], 'chunk_size': chunk_size})
        turn_data_dict = deepcopy(time_dict)
        if name == 'train':
            t_dict = deepcopy(time_dict)
            for r in routine.time_list_dict:
                t_dict.update(**r)
                time_list_dict.append(t_dict)

        # if 'id' not in df_to_process.columns:
        #     df_to_process = df_to_process.reset_index(drop=True)
        #     df_to_process['id'] = df_to_process.index
        # check if removable
        a = datetime.now()
        res_dict = routine.get_processed_data(df_to_process, batch_size=chunk_size)
        time_dict.update(phase='embedding-generation', time=(datetime.now() - a).total_seconds())
        time_list_dict.append(time_dict.copy())

        turn_data_dict.update(**res_dict)
        turn_data_dict.pop('left_emb')
        turn_data_dict.pop('right_emb')

        a = datetime.now()
        res = routine.get_word_pairs(df_to_process, res_dict)
        time_dict.update(phase='word-pairing', time=(datetime.now() - a).total_seconds())
        time_list_dict.append(time_dict.copy())
        word_pairs, emb_pairs = res
        turn_data_dict['word_pairs'] = word_pairs
        turn_data_dict['emb_pairs'] = emb_pairs
        dataset_data_dict[name] = turn_data_dict

    routine.features_dict = {}
    for name, df_to_process in to_iter:
        time_dict.update(**{'df_name': dataset_name, 'df_split': name,
               'size': df_to_process.shape[0], 'chunk_size': chunk_size})

        routine.verbose = True
        if name == 'train':
            a = datetime.now()
            _ = routine.net_train(train_word_pairs=dataset_data_dict['train']['word_pairs'],
                                  train_emb_pairs=dataset_data_dict['train']['emb_pairs'],
                                  valid_word_pairs=dataset_data_dict['valid']['word_pairs'],
                                  valid_emb_pairs=dataset_data_dict['valid']['emb_pairs'])
            # lr=2e-5, num_epochs=25) # batch_size=128, lr=1e-5
            time_dict.update(phase='train_relevance_net', time=(datetime.now() - a).total_seconds())
            time_list_dict.append(time_dict.copy())

        args = dataset_data_dict[name]['word_pairs'], dataset_data_dict[name]['emb_pairs']
        a = datetime.now()
        word_relevance = routine.relevance_score(*args)
        time_dict.update(phase='relevance_scores', time=(datetime.now() - a).total_seconds())
        time_list_dict.append(time_dict.copy())

        a = datetime.now()
        features = routine.extract_features(word_relevance)
        time_dict.update(phase='feature_extraction', time=(datetime.now() - a).total_seconds())
        time_list_dict.append(time_dict.copy())

        routine.features_dict[name] = features
        dataset_data_dict[name]['features'] = features
        dataset_data_dict[name]['word_relevance'] = word_relevance


    for name, df_to_process in to_iter:
        time_dict.update(**{'df_name': dataset_name, 'df_split': name,
                            'size': df_to_process.shape[0], 'chunk_size': chunk_size})
        if name == 'train':
            a = datetime.now()
            routine.EM_modelling()
            time_dict.update(phase='EM_modelling', time=(datetime.now() - a).total_seconds())
            time_list_dict.append(time_dict.copy())
            tmp_dict = deepcopy(time_dict)
            tmp_dict['EM_modelling'] = True
            for row in routine.timing_models.to_dict(orient='row'):
                tmp_dict.update(**row)
                time_list_dict.append(tmp_dict.copy())

        lr = True
        reload = False
        a = datetime.now()
        if lr:
            match_score = routine.get_match_score(dataset_data_dict[name]['features'], lr=lr, reload=reload)
        else:
            match_score = routine.get_match_score(dataset_data_dict[name]['features'], reload=reload)
        time_dict.update(phase='final_match_score', time=(datetime.now() - a).total_seconds())
        time_list_dict.append(time_dict.copy())
        dataset_data_dict[name]['match_score'] = match_score

        word_relevance, features = dataset_data_dict[name]['word_relevance'], dataset_data_dict[name]['features']
        a = datetime.now()
        word_relevance = routine.get_contribution_score(word_relevance, features, lr=lr, reload=True)
        time_dict.update(phase='feature_contribution', time=(datetime.now() - a).total_seconds())
        time_list_dict.append(time_dict.copy())
        dataset_data_dict[name]['word_relevance'] = word_relevance
        dataset_data_dict[name].pop('emb_pairs')

    return time_list_dict, dataset_data_dict
        # match_score, data_dict, res, features, word_relevance

if __name__ == '__main__':
    prefix = '/home/baraldian'
    softlab_path = os.path.join(prefix + '/content/drive/Shareddrives/SoftLab/')
    project_path = os.path.join(softlab_path, 'Projects', 'WYM')
    sys.path.append(os.path.join(project_path))
    sys.path = list(set(sys.path))
    reset_files = False  # @param {type:"boolean"}
    reset_networks = False  # @param {type:"boolean"}
    we_finetuned = "sBERT"  # @param ["False", "sBERT"]
    sentence_embedding = False  # @param {type:"boolean"}
    model_name = "BERT"  # @param ["BERT"]
    chunk_size = 128  # @param {type:"slider", min:4, max:512, step:4}
    if we_finetuned == 'False':
        we_finetuned = False

    time_list_dict = []
    for dataset_name in tqdm(sorted_dataset_names):
        dataset_path = os.path.join(softlab_path, 'Dataset', 'Entity Matching', dataset_name)
        model_files_path = os.path.join(project_path, 'dataset_files', dataset_name, model_name)
        common_files = os.path.join(project_path, 'dataset_files', 'Abt-Buy')
        results_path = os.path.join(model_files_path, 'results')
        try:
            os.makedirs(model_files_path)
        except:
            pass
        try:
            os.makedirs(results_path)
        except:
            pass

        routine = Routine(dataset_name, dataset_path, project_path, reset_files=reset_files,
                          reset_networks=reset_networks, softlab_path=softlab_path, model_name=model_name,
                          we_finetuned=we_finetuned, we_finetune_path=None,
                          sentence_embedding=sentence_embedding, train_batch_size=16
                          )
        turn_time, dataset_data_dict = run_pipeline(dataset_name=dataset_name, routine=routine)
        time_list_dict += turn_time

        suffix = '_last'
        tmp_path = os.path.join(results_path, f'timing{suffix}.pickle')
        with open(tmp_path, 'wb') as file:
            pickle.dump(turn_time, file)

        tmp_path = os.path.join(results_path, f'data_dict{suffix}.pickle')
        with open(tmp_path, 'wb') as file:
            pickle.dump(dataset_data_dict, file)

    tmp_path = os.path.join(project_path, 'dataset_files', f'all_timing{suffix}.csv')
    pd.DataFrame(time_list_dict).to_csv(tmp_path)
