import torch.nn as nn

class DNN(nn.Module):
    """"
    Simple Neural Network with one input, one hidden, and one output layer
    """
    def __init__(self, params):
        super(DNN, self).__init__()
        self.classif = nn.Sequential(
            nn.Linear(params["input_dim"],params["hidden_size"]),
            nn.ReLU(),
            nn.Linear(params["hidden_size"], params["hidden_size"]),
            nn.ReLU(),
            nn.Dropout(params["drop_prob"]),
            nn.Linear(params["hidden_size"],params["label_dim"]))

    def forward(self, x):
        x = self.classif(x)
        return x
