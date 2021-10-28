from torch import nn

# 5 hidden layers
class Model2(nn.Module):
    def __init__(self, input_shape, hidden_size):
        super(Model2, self).__init__()
        self.input_layer = nn.Linear(input_shape, hidden_size)
        self.h1 = nn.Linear(hidden_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, hidden_size)
        self.h3 = nn.Linear(hidden_size, hidden_size)
        self.h4 = nn.Linear(hidden_size, hidden_size)
        self.h5 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.input_layer(inputs))
        x = self.relu(self.h1(x))
        x = self.relu(self.h2(x))
        x = self.relu(self.h3(x))
        x = self.relu(self.h4(x))
        x = self.relu(self.h5(x))
        x = nn.sigmoid(self.output_layer(x))
        return x
