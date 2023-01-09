import gc
import logging
import math
import os

import torch
from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader


def create_sentence(ds):
    # Define columns
    label_col = 'label'
    ignore_col = ['left_id', 'right_id']
    left_col = [x for x in ds.columns if x.startswith('left_') and x not in ignore_col]
    right_col = [x for x in ds.columns if x.startswith('right_') and x not in ignore_col]

    # Create sentence left and right
    sent_left = ds[left_col].apply(lambda x: ' '.join(x.values.astype(str)), axis=1).values
    sent_right = ds[right_col].apply(lambda x: ' '.join(x.values.astype(str)), axis=1).values

    # Create result list
    sent_list = [{'left': a, 'right': b, 'label': label} for a, b, label in
                 zip(sent_left, sent_right, ds[label_col].values)]

    return sent_list


def create_cosine_input_format(sentences):
    samples = []
    for row in sentences:
        inp_example = InputExample(texts=[row['left'], row['right']], label=row['label'])
        samples.append(inp_example)
    return samples


def finetune_BERT(routine, num_epochs=10, model_save_path=None, train_batch_size=32):
    sent_train = create_sentence(routine.train_merged.copy())
    sent_valid = create_sentence(routine.valid_merged.copy())
    sent_test = create_sentence(routine.test_merged.copy())
    train_samples = create_cosine_input_format(sent_train)
    valid_samples = create_cosine_input_format(sent_valid)
    test_samples = create_cosine_input_format(sent_test)
    model_name = 'bert-base-uncased'

    # Read the dataset
    if model_save_path is None:
        model_save_path = os.path.join(routine.model_files_path, 'sBERT')
    model_args = {'output_hidden_states': True, 'output_attentions': True}
    word_embedding_model = models.Transformer(model_name, model_args=model_args)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.ContrastiveLoss(model=model)
    logging.basicConfig(level=logging.INFO)
    logging.info("Read valid dataset")
    evaluator = BinaryClassificationEvaluator.from_input_examples(valid_samples, name='em-dev')
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))
    torch.cuda.empty_cache()

    if hasattr(routine, 'we'):
        del routine.we
    gc.collect()
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              output_path=model_save_path,
              checkpoint_path= model_save_path,
              show_progress_bar=True,
              )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    # assert False
    return model_save_path


def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert (isinstance(size, torch.Size))
    return " × ".join(map(str, size))


def dump_tensors(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    import gc
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print("%s:%s%s %s" % (type(obj).__name__,
                                          " GPU" if obj.is_cuda else "",
                                          " pinned" if obj.is_pinned else "",
                                          pretty_size(obj.size())))
                    total_size += obj.numel()
                del obj
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    print("%s → %s:%s%s%s%s %s" % (type(obj).__name__,
                                                   type(obj.data).__name__,
                                                   " GPU" if obj.is_cuda else "",
                                                   " pinned" if obj.data.is_pinned else "",
                                                   " grad" if obj.requires_grad else "",
                                                   " volatile" if obj.volatile else "",
                                                   pretty_size(obj.data.size())))
                    total_size += obj.data.numel()
                    del obj
        except Exception as e:
            pass
    print("Total size:", total_size)
