import os
import pickle
from unittest import TestCase

import numpy as np
import pandas as pd
import torch

from WordPairGenerator import WordPairGenerator


class WordEmbeddingFake():
    def __init__(self, device='auto'):
        pass

    def get_word_embeddings(self, words):
        words_list = words.split()
        word_embedding = np.random.random([len(words_list), 50])

        return torch.tensor(word_embedding), words_list


class TestWordPairGenerator(TestCase):

    def test_process_df(self):
        we = WordEmbeddingFake()
        words_pairs_dict, emb_pairs_dict = {}, {}
        train = pd.read_csv(os.path.join('G:\\Drive condivisi\\SoftLab\\Dataset\\Entity Matching\\BeerAdvo-RateBeer',
                                         'train_merged.csv'))
        self.model_files_path = os.path.join(
            'G:\\Drive condivisi\\SoftLab\\Projects\\Concept level EM (exclusive-inclluse words)\\dataset_files\\BeerAdvo-RateBeer\\wym\\')
        self.embeddings = {}
        self.words = {}
        for df_name in ['table_A', 'table_B']:
            tmp_path = os.path.join(self.model_files_path, 'emb_' + df_name + '.csv')
            with open(tmp_path, 'rb') as file:
                self.embeddings[df_name] = pickle.load(file)
            tmp_path = os.path.join(self.model_files_path, 'words_list_' + df_name + '.csv')
            with open(tmp_path, 'rb') as file:
                self.words[df_name] = pickle.load(file)

        word_pair_generator = WordPairGenerator(self.words, self.embeddings, df=train)

        word_pairs, emb_pairs = word_pair_generator.process_df(train.sample(100))

        df = pd.DataFrame(word_pairs)

        self.assertIsInstance(df, pd.DataFrame)

    def test_stable_marriage(self):
        A = np.array([[0, 2, 1, 3, 4],
                      [4, 2, 1, 0, 3],
                      [1, 4, 2, 3, 0]])
        B = np.array([[1, 0, 2],
                      [1, 2, 0],
                      [0, 1, 2],
                      [0, 2, 1],
                      [2, 0, 1]])
        res = WordPairGenerator.stable_marriage(A, B)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(res.shape == (3, 2))

    def test_most_similar_pairs(self):
        sim_mat = np.array([[.2, .3, .6],
                            [.5, .2, .7],
                            [.2, .1, .15],
                            [.5, .6, .3]])
        pairs, sim = WordPairGenerator.most_similar_pairs(sim_mat=sim_mat)
        self.assertTrue(pairs.shape == (4, 2))
