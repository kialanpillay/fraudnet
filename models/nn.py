import torch.nn as nn
import torch.nn.functional as F


class Model1(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_norm=False):
        super(Model1, self).__init__()
        self.batch_norm = batch_norm
        if batch_norm:
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
        else:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
            self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if self.batch_norm:
            return self.layers(x)
        else:
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.sigmoid(self.output_layer(x))
            return x


class Model2(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Model2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.sigmoid(self.output_layer(x))
        return x


class Model3(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Model3, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.sigmoid(self.output_layer(x))
        return x
