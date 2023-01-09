from unittest import TestCase
import pandas as pd
from FeatureExtractor import FeatureExtractor
import os

class TestFeatureExtractor(TestCase):
    def test_extract_features(self):
        train = pd.read_csv(
            os.path.join('G:\\Drive condivisi\\SoftLab\\Dataset\\Entity Matching\\DBLP-ACM', 'train_merged.csv'))
        test_path = 'G:\\Drive condivisi\\SoftLab\\Projects\\Concept level EM (exclusive-inclluse words)\\dataset_files\\DBLP-ACM\\wym\\test_files'

        tmp_path = os.path.join(test_path, 'word_pair_corrected.csv')
        word_pair_corrected = pd.read_csv(tmp_path)

        features = FeatureExtractor().extract_features(word_pair_corrected, complementary=False)

        self.assertIsInstance(features, pd.DataFrame)
