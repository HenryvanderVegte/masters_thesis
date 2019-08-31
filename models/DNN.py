import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, params):
        super(DNN, self).__init__()
        self.classif = nn.Sequential(
            nn.Linear(params["input_dim"],params["hidden_size"]),
            nn.ReLU(),
            nn.Dropout(params["drop_prob"]),
            nn.Linear(params["hidden_size"],params["label_dim"]))

    def forward(self, x):
        x = self.classif(x)
        return x
