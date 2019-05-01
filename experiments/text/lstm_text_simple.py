import torch
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from experiments.util.data_loader import *
from nltk.metrics import ConfusionMatrix, accuracy
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
labels_count = len(set(list(label_to_id.values())))

TRAIN_EMBEDDINGS_LABELS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//embeddings//2-dim//train_embeddings.txt"
TRAIN_EMBEDDINGS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//embeddings//2-dim//train_embeddings.npy"
label_dict, feature_dict = load_dict_from_binary(TRAIN_EMBEDDINGS_LABELS, TRAIN_EMBEDDINGS, label_to_id)

DEV_EMBEDDINGS_LABELS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//embeddings//2-dim//dev_embeddings.txt"
DEV_EMBEDDINGS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//embeddings//2-dim//dev_embeddings.npy"
dev_label_dict, dev_feature_dict = load_dict_from_binary(DEV_EMBEDDINGS_LABELS, DEV_EMBEDDINGS, label_to_id)



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
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers)

        # dropout layer
        #self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)

        lstm_out, _ = self.lstm(x)

        # dropout and fully-connected layer
        #out = self.dropout(lstm_out)
        out = self.fc(lstm_out)

        out = out[-1]

        # sigmoid function
        sig_out = self.sig(out)

        # return last sigmoid output and hidden state
        return sig_out

embedding_dim = 300
hidden_dim = 50
n_layers = 2
batch_size = 1
clip=5

train_dataset = prepare_data(feature_dict, label_dict)
train_loader = utils.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

dev_dataset = prepare_data(dev_feature_dict, dev_label_dict)
dev_loader = utils.DataLoader(dev_dataset, shuffle=False, batch_size=batch_size)

net = SentimentLSTM(labels_count, embedding_dim, hidden_dim, n_layers).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.1, momentum=0.9)

# Train the model
total_step = len(train_loader)

for e in range(15):
    # batch loop
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)

        #transpose from [batch_size, time_steps, dim] to [time_steps, batch_size, dim]
        inputs = torch.transpose(inputs, 0, 1)

        labels = labels.to(device, dtype=torch.int64).view(-1)

        net.zero_grad()

        output = net(inputs)

        loss = criterion(output, labels)
        loss.backward()

        optimizer.step()


        if (i + 1) % 20 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(e + 1, 30, i + 1, total_step, loss.item()))

        predictions = []
        gold = []
        with torch.no_grad():
            for inputs, labels in dev_loader:
                inputs = inputs.to(device)
                inputs = torch.transpose(inputs, 0, 1)
                labels = labels.to(device, dtype=torch.int64).view(-1)

                output = net(inputs)

                _, predicted = torch.max(output.data, 1)
                predictions += predicted.data.tolist()
                gold += labels.data.tolist()
        print("Accuracy: " + str(accuracy(gold, predictions)))
        cm = ConfusionMatrix(gold, predictions)
        print("Confusion matrix:\n" + str(cm))

