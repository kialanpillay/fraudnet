import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import balanced_accuracy_score


def validate(model, loader, save=False):
    """
    Computes evaluation metrics for a PyTorch model

    Parameters
    ----------
    model: models.nn.FeedForwardNeuralNetwork
        Trained feed-forward neural network model
    loader: torch.utils.data.DataLoader
        Iterable dataloader
    save: float, optional
        Save model predictions to .csv

    Returns
    -------
    dict
    """
    criterion = nn.BCELoss()
    val_loss = 0.0
    val_steps = 0
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            outputs = model(inputs)
            predictions = [1 if o >= 0.5 else 0 for o in outputs.numpy()]
            y_true.append(targets.numpy())
            y_pred.append(predictions)

            loss = criterion(outputs, targets)
            val_loss += loss.cpu().numpy()
            val_steps += 1

    loss = val_loss / val_steps
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    if save:
        pd.DataFrame(y_pred, columns=['Predictions']).to_csv('predictions.csv', index=False)
    return loss, {'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)}


def validate_baseline(clf, X, y):
    """
    Computes evaluation metrics for a baseline classifier

    Parameters
    ----------
    clf: baseline.NaiveClassifier, sklearn.svm.LinearSVC
        Fitted classifier
    X: numpy.ndarray
        Features
    y: float, optional
        Class labels

    Returns
    -------
    dict
    """
    return {'balanced_accuracy': balanced_accuracy_score(y, clf.predict(X))}
