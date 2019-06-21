import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DNN, self).__init__()
        self.classif = nn.Sequential(
            nn.Linear(input_dim,32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32,num_classes))

    def forward(self, x):
        x = self.classif(x)
        return x
