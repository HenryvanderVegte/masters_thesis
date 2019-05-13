import torch
import torch.utils.data as utils
import torch.nn as nn
import torch.optim as optim
import os
from classification.util.experiments_util import log_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SentimentRNN(nn.Module):
    def __init__(self, output_size, embedding_dim, n_layers):
        super().__init__()

        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=n_layers, dropout=0.4)

        self.fc = nn.Linear(embedding_dim, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[-1]
        out = self.fc(out)
        out = self.relu(out)
        return out

def train(train_dataset, dev_dataset, experiment_path, label_to_id, logger, params):
    train_loader = utils.DataLoader(train_dataset, shuffle=False, batch_size=params["batch_size"])
    dev_loader = utils.DataLoader(dev_dataset, shuffle=False, batch_size=params["batch_size"])

    labels_count = len(set(list(label_to_id.values())))
    seq_length = train_loader.dataset[0][0].size()[0]
    embedding_dim = train_loader.dataset[0][0].size()[1]
    n_layers = params["layers"]

    model = SentimentRNN(labels_count, embedding_dim, n_layers).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    total_step = len(train_loader)

    for e in range(params["epochs"]):
        # batch loop
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)

            # transpose from [batch_size, time_steps, dim] to [time_steps, batch_size, dim]
            inputs = torch.transpose(inputs, 0, 1)

            labels = labels.to(device, dtype=torch.int64).view(-1)

            model.zero_grad()

            output = model(inputs)

            loss = criterion(output, labels)
            loss.backward()

            optimizer.step()

        train_loss = 0
        with torch.no_grad():
            for inputs, labels in dev_loader:
                inputs = inputs.to(device)
                inputs = torch.transpose(inputs, 0, 1)
                labels = labels.to(device, dtype=torch.int64).view(-1)

                output = model(inputs)
                train_loss = criterion(output, labels).item()

        test_loss = 0
        with torch.no_grad():
            for inputs, labels in dev_loader:
                inputs = inputs.to(device)
                inputs = torch.transpose(inputs, 0, 1)
                labels = labels.to(device, dtype=torch.int64).view(-1)

                output = model(inputs)
                test_loss = criterion(output, labels).item()

        print('{}; {:.0f} {:.4f}; {:.4f}'.format(e + 1, params["epochs"], train_loss, test_loss))

        # export pytorch model
        if (e+1) % 10 == 0:
            # export pytorch model
            epoch_path = os.path.join(experiment_path, "epoch_" + str(e))
            os.mkdir(epoch_path)
            model_path = os.path.join(epoch_path,  "rnn.pth")
            torch.save(model.state_dict(), model_path)

            onnx_model_path = os.path.join(epoch_path,  "rnn.onnx")
            dummy_input = torch.randn(1, seq_length, embedding_dim, device='cuda')
            torch.onnx.export(model, dummy_input, onnx_model_path, verbose=False)

    model_path = os.path.join(experiment_path, "rnn.pth")
    torch.save(model.state_dict(), model_path)

    onnx_model_path = os.path.join(experiment_path, "rnn.onnx")
    dummy_input = torch.randn(1, seq_length, embedding_dim, device='cuda')
    torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True)


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
            inputs = torch.transpose(inputs, 0, 1)
            labels = labels.to(device, dtype=torch.int64).view(-1)

            output = model(inputs)

            _, predicted = torch.max(output.data, 1)
            predictions += predicted.data.tolist()
            gold += labels.data.tolist()

    log_metrics(gold, predictions, logger)