from torch import nn


# 3 hidden layers - variable parameters
class Model1(nn.Module):
    def __init__(self, input_shape, hidden_size):
        super(Model1, self).__init__()
        self.input_layer = nn.Linear(input_shape, hidden_size)
        self.h1 = nn.Linear(hidden_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, hidden_size)
        self.h3 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        # self.batchnorm1 = nn.BatchNorm1d(64)
        # self.batchnorm2 = nn.BatchNorm1d(64)
        # self.batchnorm3 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.input_layer(inputs))
        # x = self.batchnorm1(x)
        x = self.relu(self.h1(x))
        # x = self.batchnorm2(x)
        x = self.relu(self.h2(x))
        # x = self.batchnorm3(x)
        x = self.relu(self.h3(x))
        x = self.sigmoid(self.output_layer(x))
        return x
