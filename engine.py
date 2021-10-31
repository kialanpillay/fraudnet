import torch
import torch.nn as nn
import torch.utils.data
from ray import tune

from evaluation import validate
from models.nn import FeedForwardNeuralNetwork
from preprocessing import get_data_loaders


def train(config, filepath='./data/creditcard.csv', verbose=False):
    model = FeedForwardNeuralNetwork(config['input_dim'], config['hidden_dim'], config['hidden_layers'],
                                     config['batch_norm'])

    criterion = nn.BCELoss()

    train_loader, val_loader, _ = get_data_loaders(filepath, config['batch_size'])

    optimizer = torch.optim.RMSprop(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    for epoch in range(config['num_epochs']):
        if verbose:
            print('Epoch {:2d}'.format(epoch + 1))
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if i % 500 == 0 and verbose:
                print('Iteration {:<4d}  | Loss: {:5.6f}'.format(i, loss.item()))

        val_loss, metrics = validate(model, val_loader)
        if verbose:
            print("\nValidation Set Performance")
            print('{:<15s} : {:5.6f}'.format("Loss", val_loss)),
            print('{:<15s} : {:5.6f}'.format("Balanced Acc.", metrics['balanced_accuracy']))

        if config['hyper_opt']:
            tune.report(loss=val_loss, balanced_accuracy=metrics['balanced_accuracy'])
