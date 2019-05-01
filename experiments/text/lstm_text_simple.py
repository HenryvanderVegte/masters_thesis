import torch
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from experiments.util.data_loader import *
import gensim
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

label_to_id = {
    "hap":"0",
    "exc":"0",
    "sad":"1",
    "ang":"2",
    "neu":"3",
}

TEST_EMBEDDINGS_LABELS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//embeddings//2-dim//test_embeddings.txt"
TEST_EMBEDDINGS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//embeddings//2-dim//test_embeddings.npy"
label_dict, feature_dict = load_dict_from_binary(TEST_EMBEDDINGS_LABELS, TEST_EMBEDDINGS, label_to_id)

labels_count = len(set(list(label_to_id.values())))

def prepare_data(feature_dict, label_dict, seq_length = 50):
    fl = []
    labels = []
    for key in feature_dict.keys():
        fl.append(torch.stack([torch.Tensor(i) for i in feature_dict[key][:seq_length,:]]))
        labels.append(label_dict[key])

    labels = np.array(labels).reshape(-1,1)
    padded_features = pad_sequence(fl)

    labels = torch.stack([torch.Tensor(i) for i in labels])

    padded_features = torch.transpose(padded_features, 0, 1)
    dataset = utils.TensorDataset(padded_features, labels)

    return dataset

class SentimentLSTM(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.1):
        """
        Initialize the model by setting up the layers.
        """
        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        # dropout layer
        #self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)

        lstm_out, hidden = self.lstm(x, hidden)

        # dropout and fully-connected layer
        #out = self.dropout(lstm_out)
        out = self.fc(lstm_out)

        out = out[:,-1]

        # sigmoid function
        sig_out = self.sig(out)

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

embedding_dim = 300
hidden_dim = 200
n_layers = 5
batch_size = 1
clip=5
dataset = prepare_data(feature_dict, label_dict)

train_loader = utils.DataLoader(dataset, shuffle=False, batch_size=batch_size)

net = SentimentLSTM(labels_count, embedding_dim, hidden_dim, n_layers).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)

# Train the model
total_step = len(train_loader)

counter = 0

print_every = 100

for e in range(30):
    # initialize hidden state
    h = net.init_hidden(batch_size)
    total_step = len(train_loader)
    # batch loop
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)

        #transpose from [batch_size, time_steps, dim] to [time_steps, batch_size, dim]
        #inputs = torch.transpose(inputs, 0, 1)

        labels = labels.to(device, dtype=torch.int64).view(-1)

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output, labels)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                        .format(e + 1, 30, i + 1, total_step, loss.item()))