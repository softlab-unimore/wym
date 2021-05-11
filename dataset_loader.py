from sklearn.preprocessing import Normalizer,StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.pipeline import Pipeline




class TanhScaler(StandardScaler):
  def __init__(self, scale_factor=1.):
    super().__init__()
    self.scale_factor=scale_factor

  def transform(self, X, copy=None):
    tmp = super().transform(X)
    return 0.5 * (np.tanh( self.scale_factor * tmp) + 1)


class DatasetSpaiate(Dataset):
    def __init__(self, df, word_vectors):
        X = self.preprocess(df, word_vectors)
        self.X = X
        self.y = torch.tensor(self.aggregated.label.to_numpy(), dtype=torch.float).view([-1, 1])

    def preprocess(self, df, word_vectors):
        grouped = df.groupby(['attribute', 'word'], as_index=False).agg({'label': ['mean']}).droplevel(1, 1)
        vectors = word_vectors[grouped.word.values]
        labels = grouped['label'].to_numpy()
        if hasattr(self, 'scaler') == False:
            self.scaler = Pipeline([('tanh', TanhScaler(scale_factor=0.1)), ('std', StandardScaler())]).fit(vectors)
            self.label_scaler = Pipeline([('std', StandardScaler()), ('minMax',MinMaxScaler())]).fit(labels.reshape(-1, 1))
        # attr_dummies = pd.get_dummies(grouped.attribute, prefix='attr')
        # X_train = np.hstack([attr_dummies.to_numpy(), self.scaler.transform(vectors)])
        X_train = self.scaler.transform(vectors)
        grouped['label'] = self.label_scaler.transform(labels.reshape(-1, 1))
        self.aggregated = grouped
        return torch.tensor(X_train)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item: int):
        return self.X[item], self.y[item]

class DatasetAccoppiate(Dataset):
    def __init__(self, df, word_vectors):
        X = self.preprocess(df, word_vectors)
        self.X = X
        self.y = torch.tensor(self.aggregated.label.values, dtype=torch.float).view([-1, 1])

    def preprocess(self, df, word_vectors):
        grouped = df.groupby(['attribute', 'left_word', 'right_word'], as_index=False).agg(
            {'label': ['mean']}).droplevel(1, 1)
        labels = grouped['label'].to_numpy()
        vectors_l = word_vectors[grouped.left_word.values]
        vectors_r = word_vectors[grouped.right_word.values]
        stacked = np.stack([vectors_l, vectors_r])
        mean_vectors = stacked.mean(axis=0)
        abs_diff = np.abs(np.diff(stacked, axis=0)).squeeze()
        if hasattr(self, 'tanh_scaler_mean') == False:
            self.tanh_scaler_mean, self.tanh_scaler_diff = TanhScaler().fit(mean_vectors), TanhScaler().fit(
                abs_diff)
            self.label_scaler = Pipeline([('std', StandardScaler()), ('minMax', MinMaxScaler())]).fit(
                labels.reshape(-1, 1))
        # attr_dummies = pd.get_dummies(grouped.attribute, prefix='attr')
        # X = np.hstack([attr_dummies, self.tanh_scaler_mean.transform(mean_vectors), self.tanh_scaler_diff.transform(abs_diff)])
        X = np.hstack([self.tanh_scaler_mean.transform(mean_vectors), self.tanh_scaler_diff.transform(abs_diff)])
        grouped['label'] = self.label_scaler.transform(labels.reshape(-1, 1))
        self.aggregated = grouped
        return torch.tensor(X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item: int):
        return self.X[item], self.y[item]



# Final dataset loader
class FinalLoader(Dataset):
    def __init__(self, df, exclude_attrs):
        self.X = torch.tensor(df[np.setdiff1d(df.columns, exclude_attrs)].to_numpy(), dtype=torch.float)
        self.y = torch.tensor(df['label'], dtype=torch.float).view([-1, 1])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item: int):
        return self.X[item], self.y[item]
