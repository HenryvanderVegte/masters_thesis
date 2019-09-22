import torch
import torch.nn as nn

"""
LSTM model, creates an n-layer LSTM with a final fully connected layer
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.hidden_size = params["hidden_size"]
        self.n_layers = params["layers"]
        self.rnn = nn.LSTM(input_size=params["input_dim"], hidden_size=params["hidden_size"], num_layers=params["layers"], dropout=params["drop_prob"], batch_first=True)

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        self.dropout = nn.Dropout(params["fully_connected_drop_prob"])

        self.fc = nn.Linear(params["hidden_size"], params["label_dim"])
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.relu = nn.ReLU()

    def forward(self, x, lengths, hidden):
        batch_size = x.shape[0]

        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)

        packed, hidden = self.rnn(x, hidden)

        hidden_activation, len = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
        last_out = torch.empty(batch_size, self.hidden_size, dtype=torch.float, device=device)

        for j, x in enumerate(len):
            last_out[j,:] = hidden_activation[j,(x-1),:]

        out = self.dropout(last_out)

        out = self.fc(out)
        out = self.relu(out)

        return out, hidden, last_out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        if str(device) == 'cuda':
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_size).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_size).zero_())

        return hidden
