import torch.nn as nn


class FeedForwardNeuralNetwork(nn.Module):
    """
    Feed-forward neural network model for binary classification. Supports batch normalisation layers
    """
    def __init__(self, input_dim, hidden_dim=64, hidden_layers=3, batch_norm=False):
        super(FeedForwardNeuralNetwork, self).__init__()
        self.num_layers = hidden_layers
        self.batch_norm = batch_norm
        if batch_norm:
            modules = [nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()]
            for i in range(hidden_layers - 1):
                modules.append(nn.Linear(hidden_dim, hidden_dim))
                modules.append(nn.BatchNorm1d(hidden_dim))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(hidden_dim, 1))
            modules.append(nn.Sigmoid())
            self.layers = nn.Sequential(*modules)
        else:
            modules = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            for i in range(hidden_layers - 1):
                modules.append(nn.Linear(hidden_dim, hidden_dim))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(hidden_dim, 1))
            modules.append(nn.Sigmoid())
            self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)
