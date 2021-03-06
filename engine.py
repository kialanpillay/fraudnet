import torch
import torch.nn as nn
import torch.utils.data
from ray import tune

from evaluation import validate
from models.nn import FeedForwardNeuralNetwork
from preprocessing import get_data_loaders
import matplotlib.pyplot as plt


def train(config, filepath='./data/creditcard.csv', verbose=False):
    """
    Train a PyTorch model

    Parameters
    ----------
    config : dict
       Training configuration
    filepath: str, optional
        Dataset filepath
    verbose: bool, optional
        Print per-epoch metrics

    Returns
    -------
    FeedForwardNeuralNetwork
    """
    model = FeedForwardNeuralNetwork(config['input_dim'], config['hidden_dim'], config['hidden_layers'],
                                     config['batch_norm'])

    criterion = nn.BCELoss()

    train_loader, val_loader, _ = get_data_loaders(filepath, config['batch_size'])

    optimizer = torch.optim.RMSprop(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    train_losses = []
    val_losses = []
    for epoch in range(config['num_epochs']):
        train_loss = 0.0
        train_steps = 0
        if verbose:
            print('Epoch {:2d}'.format(epoch + 1))
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

            if i % 500 == 0 and verbose:
                print('Iteration {:<4d}  | Loss: {:5.6f}'.format(i, loss.item()))

        if verbose:
            print("\nTrain Set Performance")
            print('{:<15s} : {:5.6f}'.format("Loss", train_loss / train_steps)),

        loss = train_loss / train_steps
        train_losses.append(loss)

        val_loss, metrics = validate(model, val_loader)
        val_losses.append(val_loss)
        if verbose:
            print("\nValidation Set Performance")
            print('{:<15s} : {:5.6f}'.format("Loss", val_loss)),
            print('{:<15s} : {:5.6f}'.format("Balanced Acc.", metrics['balanced_accuracy']))
            print()

        if config['hyper_opt']:
            tune.report(loss=val_loss, balanced_accuracy=metrics['balanced_accuracy'])

    if not verbose and not config['hyper_opt']:
        print("\nValidation Set Performance ({} Epochs)".format(config['num_epochs']))
        print('{:<15s} : {:5.6f}'.format("Loss", val_loss)),
        print('{:<15s} : {:5.6f}'.format("Balanced Acc.", metrics['balanced_accuracy']))

    if config['graph']:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train Set Loss")
        plt.plot(val_losses, label="Validation Set Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Binary Cross-Entropy Loss")
        plt.legend()
        plt.xlim([1, config['num_epochs']])
        plt.savefig('Loss.png')
        plt.show()

    if config['hyper_opt'] is not True:
        return model


