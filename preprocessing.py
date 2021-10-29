import numpy as np
import pandas as pd
import torch.utils.data
from sklearn.model_selection import train_test_split


def load(filepath='data/creditcard.csv'):
    df = pd.read_csv(filepath)
    X = df.drop(['Time', 'Class'], axis=1).values.astype(np.float32)
    y = df['Class'].values.astype(np.float32).reshape(-1, 1)
    return X, y


def partition(train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2):
    X, y = load()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                    test_size=test_ratio / (test_ratio + validation_ratio),
                                                    random_state=1)
    return X_train, y_train, X_val, y_val, X_test, y_test


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
