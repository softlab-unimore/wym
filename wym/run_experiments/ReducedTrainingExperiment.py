import os

import numpy as np


def do_reduced_training_experiment(routine, n_repetition=3, train_size_list=[500, 1000, 2000], reset=False):
    print(f'\n\n\nProcessing >>>> {routine.dataset_name}\n\n\n')
    model_name = 'wym'
    # Generate dir for reduced training
    reduced_train_general_path = os.path.join(routine.model_files_path, 'Reduced training set')
    try:
        os.mkdir(reduced_train_general_path)
    except Exception as e:
        print(e)

    for curr_train_size in train_size_list:
        print(f'Train size >>>> {curr_train_size}')
        reduced_train_k_path = os.path.join(routine.model_files_path, 'Reduced training set',
                                            'train_size_' + str(curr_train_size))
        try:
            os.mkdir(reduced_train_k_path)
        except Exception as e:
            print(e)

        # Sample training
        df = routine.train_merged
        vc = df['label'].value_counts()
        propotions = {1: vc[1] / (vc[0] + vc[1]), 0:
            vc[0] / (vc[1] + vc[0])}
        grouped = df.groupby('label')

        train_df_list = []
        for r in range(n_repetition):
            sample_list = []
            for turn_key, turn_df in grouped:
                tmp_size = int(np.floor(curr_train_size * propotions[turn_key]))
                tmp_df = turn_df.sample(tmp_size, random_state=0)
                sample_list.append(tmp_df)
            train_df_list.append(pd.concat([sample_list[0], sample_list[1]]))

        # Save current training with other files
        for i, turn_df in enumerate(train_df_list):
            turn_path = os.path.join(reduced_train_k_path, str(i))
            try:
                os.mkdir(turn_path)
            except Exception as e:
                print(e)
            try:
                os.makedirs(os.path.join(turn_path, 'results'))
            except Exception as e:
                print(e)
                pass
            turn_df.to_csv(os.path.join(turn_path, 'train_merged.csv'), index=False)
            for name, df in [('tableA.csv', routine.table_A),
                             ('tableB.csv', routine.table_B),
                             ('test_merged.csv', routine.test_merged),
                             ('valid_merged.csv', routine.valid_merged),
                             ]:
                df.to_csv(os.path.join(turn_path, name), index=False)

        # Apply the model
        for i, turn_df in enumerate(tqdm(train_df_list)):

            turn_path = os.path.join(reduced_train_k_path, str(i))

            if os.path.isfile(os.path.join(turn_path, 'results', 'performances.csv')) and reset:
                print('Already done')
                continue

            finetuned_path = os.path.join(routine.project_path, 'dataset_files', routine.dataset_name, model_name,
                                          'sBERT')
            turn_routine = Routine(routine.dataset_name, dataset_path=turn_path, project_path=routine.project_path,
                                   reset_files=True,
                                   reset_networks=True, model_files_path=turn_path,
                                   softlab_path=routine.softlab_path, model_name=model_name,
                                   # we_finetuned='SBERT',  # 0.4536 -- .5117
                                   we_finetuned=False,
                                   # we_finetune_path=finetuned_path,
                                   # num_epochs=5,
                                   sentence_embedding=False,
                                   )
            turn_routine.generate_df_embedding(chunk_size=500)
            _ = turn_routine.compute_word_pair()
            _ = turn_routine.net_train(num_epochs=50, lr=2e-5, batch_size=64);
            _ = turn_routine.preprocess_word_pairs()
            res = turn_routine.EM_modelling(do_evaluation=False)
