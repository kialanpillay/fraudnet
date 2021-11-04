import numpy as np
import pandas as pd
import torch.utils.data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_data_loaders(filepath, batch_size):
    X, y = load_data(filepath)
    X_train, y_train, X_val, y_val, X_test, y_test = partition(X, y)

    train_dataset = Dataset(X_train, y_train)
    val_dataset = Dataset(X_val, y_val)
    test_dataset = Dataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def load_data(filepath='./data/creditcard.csv'):
    df = pd.read_csv(filepath)
    X = df.drop(['Time', 'Class'], axis=1).values.astype(np.float32)
    y = df['Class'].values.astype(np.float32).reshape(-1, 1)
    return X, y


def partition(X, y, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                    test_size=test_ratio / (test_ratio + validation_ratio),
                                                    random_state=1)
    return X_train, y_train, X_val, y_val, X_test, y_test


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = StandardScaler().fit_transform(X)
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
