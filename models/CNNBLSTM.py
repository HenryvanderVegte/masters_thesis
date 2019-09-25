import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNNBLSTM(nn.Module):
    """
    Convolutional neural network with LSTM layers at the end.
    Implementation is inspired by Yang & Hirschberg 2018:
    'Predicting Arousal and Valence from Waveforms and Spectrograms using Deep Neural Networks'
    http://www.cs.columbia.edu/speech/PaperFiles/2018/yang_is18.pdf
    """

    def __init__(self, params):
        super().__init__()

        self.time_stride = 5
        self.time_kernel_size = 10
        self.conv_layer_first = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=(1, 10)),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1)))

        self.conv_layer_second = nn.Sequential(
            nn.Conv2d(20, 20, kernel_size=(10, 1)),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(self.time_kernel_size, 1), stride=(self.time_stride, 1)))

        self.adaptiveavgpool = nn.AdaptiveAvgPool2d((50, 20))

        self.hidden_size = params["hidden_size"]
        self.n_layers = params["layers"]

        self.rnn = nn.LSTM(input_size=1500, hidden_size=params["hidden_size"], num_layers=params["layers"], dropout=params["drop_prob"], batch_first=True)

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        self.classif = nn.Sequential(
            nn.Linear(20*50*20,2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048,2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048,params["labels_size"])
        )

    def forward(self, x, lengths, hidden):
        lengths = ((lengths - self.time_kernel_size) / self.time_stride) - 1
        x0 = x.unsqueeze(1)
        x1 = self.conv_layer_first(x0)

        x2 = self.conv_layer_second(x1)

        x3 = self.adaptiveavgpool(x2)

        batch_size, channels, seq_length, spectograms = x3.size()

        x3 = x3.view(batch_size, seq_length * channels * spectograms)
        out = self.classif(x3)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        if str(device) == 'cuda':
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_size).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_size).zero_())

        return hidden
