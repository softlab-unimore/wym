
from notebook_import_utility_env import *


if __name__ == '__main__':
    dataset_name = 'Abt-Buy'
    reset_files = False  # @param {type:"boolean"}
    reset_networks = False  # @param {type:"boolean"}
    we_finetuned = "False"  # @param ["False", "sBERT"]
    sentence_embedding = False  # @param {type:"boolean"}
    model_name = "BERT"  # @param ["BERT"]
    dataset_path = os.path.join(softlab_path, 'Dataset', 'Entity Matching', dataset_name)
    model_files_path = os.path.join(project_path, 'dataset_files', dataset_name, model_name)
    common_files = os.path.join(project_path, 'dataset_files', 'Abt-Buy')
    if we_finetuned == 'False':
        we_finetuned = False

    routine = Routine(dataset_name, dataset_path, project_path, reset_files=reset_files,
                      reset_networks=reset_networks, softlab_path=softlab_path, model_name=model_name,
                      we_finetuned=we_finetuned, we_finetune_path=None,
                      sentence_embedding=sentence_embedding, train_batch_size=16)
    routine.generate_df_embedding(chunk_size=100)
    _ = routine.compute_word_pair()
    predictor = routine.get_predictor()
    predictor = partial(predictor, return_data=True, lr=True)