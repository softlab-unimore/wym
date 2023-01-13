import gc
import numpy as np
import torch
from tqdm.autonotebook import tqdm
from transformers import BertModel, BertTokenizer


class WordEmbeddingFastText():

    def __init__(self, model, device='auto', verbose=False, model_path='bert-base-uncased',
                 sentence_embedding=False):
        self.sentence_embedding = sentence_embedding
        self.model = model
        # Set the device to GPU (cuda) if available, otherwise stick with CPU
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.verbose = verbose

    def get_word_embeddings(self, sentences):
        phrase_list = []
        tokens = []
        for phrase in sentences:
            words = phrase.split()
            phrase_list.append(words)
            for word in words:
                tokens.append(word)
        token_emb = self.get_token_embeddings(tokens)

        word_embeddings = []
        s = 0
        for i, phrase in enumerate(phrase_list):
            end = s + len(phrase)
            tmp_list=[]
            for x in token_emb[s:end]:
                tmp_list.append(torch.tensor(x))
            word_embeddings.append(torch.stack(tmp_list))
            s = end
        return word_embeddings, phrase_list

    def get_token_embeddings(self, token_list):

        token_vecs = self.model[token_list]
        return token_vecs

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
            words = np.concatenate([str(val).split() for val in x[not_na_mask].values])  # set of unique words here
            return ' '.join(words)
        else:
            return None

    def get_embedding_df(self, df):
        columns = np.setdiff1d(df.columns, ['id'])
        df = df.replace('None', np.nan).replace('nan', np.nan)
        sentences = df[columns].apply(WordEmbeddingFastText.get_words_to_embed, 1)
        not_None_sentences = [x for x in sentences if x is not None]
        if len(not_None_sentences) > 0:
            if self.sentence_embedding:
                tmp_emb_all, tmp_words, tmp_sentences = self.get_word_embeddings(not_None_sentences)
            else:
                tmp_emb_all, tmp_words = self.get_word_embeddings(not_None_sentences)
        emb_all, words, sentences_emb = [], [], []
        index = 0
        for i in sentences:
            if i is None:
                emb_all.append(torch.tensor([0]).to('cpu'))
                if self.sentence_embedding:
                    sentences_emb.append(torch.tensor([0]).to('cpu'))
            else:
                emb_all.append(tmp_emb_all[index].to('cpu'))
                words.append(tmp_words[index])
                if self.sentence_embedding:
                    sentences_emb.append(tmp_sentences[index].to('cpu'))
                index += 1

        emb_all = np.array(emb_all, dtype=object)

        if self.sentence_embedding:
            sentences_emb = np.array(sentences_emb, dtype=object)
            return emb_all, words, sentences_emb
        else:
            return emb_all, words

    def generate_embedding(self, df, chunk_size=500):
        emb_list, words_list, sent_emb_list = [], [], []
        n_chunk = np.ceil(df.shape[0] / chunk_size).astype(int)

        torch.cuda.empty_cache()
        if self.verbose:
            print('Computing embedding')
            to_cycle = tqdm(range(n_chunk))
        else:
            to_cycle = range(n_chunk)
        for chunk in to_cycle:
            # assert False
            if self.sentence_embedding:
                emb, words, sent_emb = self.get_embedding_df(df.iloc[chunk * chunk_size:(chunk + 1) * chunk_size])
                sent_emb_list.append(sent_emb)
            else:
                emb, words = self.get_embedding_df(df.iloc[chunk * chunk_size:(chunk + 1) * chunk_size])
            emb_list.append(emb)
            words_list += words

            gc.collect()
            torch.cuda.empty_cache()
        if len(emb_list) > 0:
            emb_list = np.concatenate(emb_list)
            if self.sentence_embedding:
                sent_emb_list = np.concatenate(sent_emb_list)
        if self.sentence_embedding:
            return emb_list, words_list, sent_emb_list
        else:
            return emb_list, words_list
