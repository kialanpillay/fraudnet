import pandas as pd


def load(filepath='data/creditcard.csv'):
    df = pd.read_csv(filepath)
    X = df.drop(['Time', 'Class'], axis=1).values
    y = df['Class'].values.reshape(-1, 1)
    return X, y
