import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SelfLearnedEmbeddingsLSTM(nn.Module):
    """
    LSTM with embedding layer before. Inputs to the embedding layers are one-hot encoded vectors with
    the dimensionality of the used vocabulary.
    """
    def __init__(self, params):
        super().__init__()

        self.hidden_size = params["hidden_size"]
        self.n_layers = params["layers"]

        self.embedding = nn.Embedding(params["vocab_size"], params["embedding_size"])
        self.rnn = nn.LSTM(input_size=params["embedding_size"], hidden_size=params["hidden_size"], num_layers=params["layers"], dropout=params["drop_prob"], batch_first=True)

        self.dropout = nn.Dropout(params["drop_prob"])

        self.fc = nn.Linear(params["hidden_size"], params["labels_size"])
        self.relu = nn.ReLU()

    def forward(self, x, lengths, hidden):
        x = x.to(device, dtype=torch.int64)
        batch_size = x.shape[0]

        out = self.embedding(x)

        out = torch.nn.utils.rnn.pack_padded_sequence(out, lengths, batch_first=True)

        out, hidden = self.rnn(out, hidden)

        pad, inp = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
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
