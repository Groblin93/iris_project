import torch.nn as nn


class nn_classification_model(nn.Module):
    def __init__(self, model):
        super(nn_classification_model, self).__init__()
        self.input_layer = nn.Linear(model.input_dim, model.lays[0])
        self.hidden_layer1 = nn.Linear(model.lays[0], model.lays[1])
        self.output_layer = nn.Linear(model.lays[1], model.output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.input_layer(x))
        out = self.relu(self.hidden_layer1(out))
        out = self.output_layer(out)
        return out
