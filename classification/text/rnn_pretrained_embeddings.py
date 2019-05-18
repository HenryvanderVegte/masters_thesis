import numpy as np
import torch
import torch.utils.data as utils
import torch.nn as nn
import torch.optim as optim
import os
from classification.util.experiments_util import log_metrics


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def __sort_by_length(inputs, labels, lengths):
    sorted_inputs = torch.stack([x for _, x in sorted(zip(lengths, inputs), key=lambda pair: pair[0], reverse=True)])
    sorted_labels = torch.stack([x for _, x in sorted(zip(lengths, labels), key=lambda pair: pair[0], reverse=True)])
    sorted_lengths = torch.stack(sorted(lengths, reverse=True))
    return sorted_inputs, sorted_labels, sorted_lengths

class SentimentRNN(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_size, n_layers, drop_prob):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=n_layers, dropout=drop_prob, batch_first=True)

        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, lengths, hidden):

        batch_size = x.shape[0]

        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)

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

def train(train_dataset, dev_dataset, experiment_path, label_to_id, logger, params):
    logger.info(str(params))

    train_loader = utils.DataLoader(train_dataset, shuffle=True, batch_size=params["batch_size"])
    dev_loader = utils.DataLoader(dev_dataset, shuffle=False, batch_size=len(dev_dataset))

    labels_count = len(set(list(label_to_id.values())))
    embedding_dim = train_loader.dataset[0][0].size()[1]
    n_layers = params["layers"]

    model = SentimentRNN(labels_count, embedding_dim, params["hidden_size"], n_layers, params["drop_prob"]).to(device)

    logger.info(model)

    # weight cross entropy loss by the label distribution
    unique, counts = np.unique(train_dataset.tensors[1], return_counts=True)
    count_dict = dict(zip(unique, counts))
    weights = 1 / np.array(list(count_dict.values()))
    weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters())

    for e in range(params["epochs"]):
        # batch loop
        h = model.init_hidden(params["batch_size"])
        train_losses = []
        for i, (inputs, labels, lengths) in enumerate(train_loader):
            inputs, labels, lengths = __sort_by_length(inputs, labels, lengths)

            if inputs.shape[0] != params["batch_size"]:
                continue
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.int64).view(-1)
            lengths = lengths.to(device, dtype=torch.int64).view(-1)
            h = tuple([each.data for each in h])

            model.zero_grad()

            output, h = model(inputs, lengths, h)

            loss = criterion(output, labels)
            loss.backward()

            train_losses.append(loss.item())
            nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

            '''
            with torch.no_grad():
                predictions = []
                gold = []
                for inputs, labels in dev_loader:
                    inputs = inputs.to(device)
                    inputs = torch.transpose(inputs, 0, 1)
                    labels = labels.to(device, dtype=torch.int64).view(-1)

                    output = model(inputs)
                    _, predicted = torch.max(output.data, 1)
                    predictions += predicted.data.tolist()
                    gold += labels.data.tolist()
                log_metrics(gold, predictions, logger)
            '''

        test_loss = 0
        predictions = []
        gold = []
        h = model.init_hidden(len(dev_dataset))
        with torch.no_grad():
            for i, (inputs, labels, lengths) in enumerate(dev_loader):
                inputs, labels, lengths = __sort_by_length(inputs, labels, lengths)
                inputs = inputs.to(device)
                labels = labels.to(device, dtype=torch.int64).view(-1)
                lengths = lengths.to(device, dtype=torch.int64).view(-1)
                h = tuple([each.data for each in h])

                output, _ = model(inputs, lengths, h)

                test_loss = criterion(output, labels)

                _, predicted = torch.max(output.data, 1)
                predictions += predicted.data.tolist()
                gold += labels.data.tolist()

        print('{}; {:.0f} {:.4f}; {:.4f}'.format(e + 1, params["epochs"], np.mean(train_losses), test_loss))

        # export pytorch model
        if (e+1) % 11 == 0:
            logger.info("Epoch nr " + str(e))
            log_metrics(gold, predictions, logger)
            # export pytorch model
            epoch_path = os.path.join(experiment_path, "epoch_" + str(e))
            os.mkdir(epoch_path)
            model_path = os.path.join(epoch_path,  "rnn.pth")
            torch.save(model.state_dict(), model_path)

        #onnx_model_path = os.path.join(epoch_path,  "rnn.onnx")
        #dummy_input = (torch.randn(1, seq_length, embedding_dim, device='cuda'), torch.randn(params["layers"], 1, params["hidden_size"], device='cuda'))
        #torch.onnx.export(model, dummy_input, onnx_model_path, verbose=False)

    model_path = os.path.join(experiment_path, "rnn.pth")
    torch.save(model.state_dict(), model_path)

    #onnx_model_path = os.path.join(experiment_path, "rnn.onnx")
    #dummy_input = torch.randn(1, seq_length, embedding_dim, device='cuda')
    #torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True)


def test(dev_dataset, experiment_path, label_to_id, logger, params):
    dev_loader = utils.DataLoader(dev_dataset, shuffle=False, batch_size=params["batch_size"])

    labels_count = len(set(list(label_to_id.values())))
    seq_length = dev_loader.dataset[0][0].size()[0]
    embedding_dim = dev_loader.dataset[0][0].size()[1]
    n_layers = params["layers"]

    model_path = os.path.join(experiment_path, "rnn.pth")
    model = SentimentRNN(labels_count, embedding_dim, n_layers).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    logger.info("Loaded model from" + model_path)

    predictions = []
    gold = []
    with torch.no_grad():
        for inputs, labels in dev_loader:
            inputs = inputs.to(device)
            #inputs = torch.transpose(inputs, 0, 1)
            labels = labels.to(device, dtype=torch.int64).view(-1)

            output = model(inputs)

            _, predicted = torch.max(output.data, 1)
            predictions += predicted.data.tolist()
            gold += labels.data.tolist()

    log_metrics(gold, predictions, logger)