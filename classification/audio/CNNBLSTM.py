import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNNBLSTM(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.time_stride = 5
        self.time_kernel_size = 10
        self.conv_layer_first = nn.Sequential(
            nn.Conv2d(1, 80, kernel_size=(1, 10)),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1)))

        self.conv_layer_second = nn.Sequential(
            nn.Conv2d(80, 80, kernel_size=(10, 1)),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(self.time_kernel_size, 1), stride=(self.time_stride, 1)))


        self.hidden_size = params["hidden_size"]
        self.n_layers = params["layers"]
        self.rnn = nn.LSTM(input_size=2400, hidden_size=params["hidden_size"], num_layers=params["layers"], dropout=params["drop_prob"], batch_first=True)

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        self.dropout = nn.Dropout(params["fully_connected_drop_prob"])

        self.fc = nn.Linear(params["hidden_size"], params["labels_size"])
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.relu = nn.ReLU()

    def forward(self, x, lengths, hidden):

        lengths = ((lengths - self.time_kernel_size) / self.time_stride) - 1
        x0 = x.unsqueeze(1)
        x1 = self.conv_layer_first(x0)

        x2 = self.conv_layer_second(x1)

        batch_size, channels, seq_length, spectograms = x2.size()

        x3 = x2.view(batch_size, seq_length, channels * spectograms)

        x = torch.nn.utils.rnn.pack_padded_sequence(x3, lengths, batch_first=True)

        packed, hidden = self.rnn(x, hidden)

        pad, inp = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
        last_out = torch.empty(batch_size, self.hidden_size, dtype=torch.float, device=device)

        for j, x in enumerate(inp):
            last_out[j,:] = pad[j,(x-1),:]

        out = self.dropout(last_out)

        out = self.fc(out)
        out = self.relu(out)
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
