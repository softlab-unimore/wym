import gc

import numpy as np
import torch
from tqdm.notebook import tqdm
from transformers import BertModel, BertTokenizer


class WordEmbedding():
    def __init__(self, device='auto', verbose=False):
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        # Set the device to GPU (cuda) if available, otherwise stick with CPU
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.verbose = verbose

    def get_word_embeddings(self, sentences):
        tokens = self.tokenizer(sentences, padding=True, return_tensors='pt')
        detected_tokens = [self.tokenizer.convert_ids_to_tokens(x) for x in tokens['input_ids']]
        token_emb = self.get_token_embeddings(tokens)
        word_token_maps, words_lists = [], []
        for x, sentence in zip(detected_tokens, sentences):
            tmp_map, tmp_words = WordEmbedding.map_token_to_word(x, sentence)
            word_token_maps.append(tmp_map)
            words_lists.append(tmp_words)
        word_embeddings = []
        for i, map in enumerate(word_token_maps):
            aggregated_emb = [torch.mean(token_emb[i, map[word_pos]], 0) for word_pos, word in
                              enumerate(words_lists[i])]
            word_embeddings.append(torch.stack(aggregated_emb))
        return word_embeddings, words_lists

    def get_token_embeddings(self, token_list):
        with torch.no_grad():
            outputs = self.model(token_list['input_ids'].to(self.device),
                                 token_list['attention_mask'].to(self.device),
                                 token_list['token_type_ids'].to(self.device))
            # Evaluating the model will return a different number of objects based on
            # how it's  configured in the `from_pretrained` call earlier. In this case,
            # becase we set `output_hidden_states = True`, the third item will be the
            # hidden states from all layers. See the documentation for more details:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)
        # token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(0, 1, 2, 3)  # layers, sentences, words, weights
        token_vecs_sum = torch.mean(token_embeddings[2:12], 0)  # I chose to use the mean instead of sum

        return token_vecs_sum

    @staticmethod
    def map_token_to_word(detected_tokens, sentence=None):
        words_map = {x: [] for x in range(len(detected_tokens))}
        words_list = []
        word_pos = -1
        for i, token in enumerate(detected_tokens[1:]):
            if token.startswith('##'):
                words_map[word_pos] += [i + 1]
                words_list[-1] += token[2:]
            else:
                word_pos += 1
                words_map[word_pos] = [i + 1]
                words_list.append(token)

        if sentence is not None and sentence != 'None':
            words_splitted = sentence.lower().split() + ['[SEP]'] # unidecode.unidecode ERROR IN accent words
            new_words_map = {x: [] for x in range(len(words_splitted))}
            word_pos = 0
            tmp_word = ''
            split_pos = 0
            while split_pos < len(words_splitted):
                tmp_word += words_list[word_pos]
                # print(f'{tmp_word} -- {words_splitted[split_pos]}, {tmp_word == words_splitted[split_pos]}')
                new_words_map[split_pos] += words_map[word_pos]
                if tmp_word == words_splitted[split_pos]:
                    split_pos += 1
                    tmp_word = ''
                word_pos += 1
            words_map = new_words_map
            words_list = words_splitted

        return words_map, words_list

    @staticmethod
    def get_words_to_embed(x):
        not_na_mask = x.notna()
        if not_na_mask.any():
            words = np.concatenate([str(val).split() for val in x[not_na_mask].values])  # set of unique words here
            return ' '.join(words)
        else:
            return None

    @staticmethod
    def get_words_by_attribute(x):
        not_na_mask = x.notna()
        if not_na_mask.any():
            words = np.concatenate([str(val).split() for val in x[not_na_mask].values] )  # set of unique words here
            return ' '.join(words)
        else:
            return None

    def get_embedding_df(self, df):
        columns = np.setdiff1d(df.columns, ['id'])
        df = df.replace('None', np.nan).replace('nan', np.nan)
        sentences = df[columns].apply(WordEmbedding.get_words_to_embed, 1)
        not_None_sentences = [x for x in sentences if x is not None]
        if len(not_None_sentences) > 0:
            tmp_emb_all, tmp_words = self.get_word_embeddings(not_None_sentences)
        emb_all, words = [], []
        index = 0
        for i in sentences:
            if i is None:
                emb_all.append(torch.tensor([0]).to(self.device))
                words.append(['[SEP]'])
            else:
                emb_all.append(tmp_emb_all[index])
                words.append(tmp_words[index])
                index += 1

        words_cutted = []
        emb_cutted = []
        for i, word_list in enumerate(words):
            last_index = word_list.index('[SEP]')
            words_cutted.append(word_list[:last_index])
            emb_cutted.append(emb_all[i][:last_index].cpu())
        emb_all = np.array(emb_cutted, dtype=object)
        return emb_all, words_cutted

    def generate_embedding(self, df, chunk_size=500):
        emb_list, words_list = [], []
        n_chunk = np.ceil(df.shape[0] / chunk_size).astype(int)
        if self.verbose:
            print('Computing embedding')
            to_cycle = tqdm(range(n_chunk))
        else:
            to_cycle = range(n_chunk)
        for chunk in to_cycle:
            gc.collect()
            torch.cuda.empty_cache()
            emb, words = self.get_embedding_df(df.iloc[chunk * chunk_size:(chunk + 1) * chunk_size])
            emb_list.append(emb)
            words_list += words
        if len(emb_list)> 0:
            emb_list = np.concatenate(emb_list)
        return emb_list, words_list
