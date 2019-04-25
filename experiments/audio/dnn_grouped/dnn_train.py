import torch
import os
import torch.nn as nn
import numpy as np
from experiments.audio.util import util
import torch.utils.data as utils
from sklearn.metrics import recall_score
from nltk.metrics import ConfusionMatrix, accuracy
import onnx


TRAIN_FILE_AUDIO = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//split//train.txt"
DEV_FILE_AUDIO = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//split//dev.txt"
EXPERIMENT_PATH = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//experiments//dnn"

label_to_id = {
    "hap":"0",
    "exc":"0",
    "sad":"1",
    "ang":"2",
    "neu":"3",
}

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 20
hidden_size = 1024
batch_size = 32
learning_rate = 0.01

class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc1_drop = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc2_drop = nn.Dropout(p=0.5)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_dim)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        #out = self.fc1_drop(out)

        out = self.fc2(out)
        out = self.relu2(out)
        #out = self.fc2_drop(out)

        out = self.fc3(out)
        return out

def train(train_file, label_to_id, experiment_path):

    train_vectors, train_labels = util.get_and_norm_train_data(train_file, label_to_id, experiment_path)

    labels_count = len(set(list(label_to_id.values())))
    instances_count = train_vectors.shape[0]
    features_count = train_vectors.shape[1]

    print("number of labels: " + str(labels_count))
    print("number of instances: " + str(instances_count))
    print("number of features: " + str(features_count))

    print("train vectors shape: " + str(train_vectors.shape))
    print("train labels shape: " + str(train_labels.shape))

    train_labels = np.array(train_labels).reshape(-1,1)

    tensor_train_x = torch.stack([torch.Tensor(i) for i in train_vectors])
    tensor_train_y = torch.stack([torch.Tensor(i) for i in train_labels])

    train_dataset = utils.TensorDataset(tensor_train_x,tensor_train_y) # create your datset
    train_dataloader = utils.DataLoader(train_dataset, batch_size=32) # create your dataloader

    model = Net(features_count, labels_count).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)

    # Train the model
    total_step = len(train_dataloader)
    print("start train")
    for epoch in range(num_epochs):
        for i, (train_vectors, labels) in enumerate(train_dataloader):
            # Move tensors to the configured device
            train_vectors = train_vectors.to(device)
            labels = labels.to(device, dtype=torch.int64).view(-1)

            # Forward pass
            outputs = model(train_vectors)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # export onnx model
    onnx_model_path = os.path.join(experiment_path, "model.onnx")
    dummy_input = torch.randn(batch_size, features_count, device='cuda')
    torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True)

    # export pytorch model
    model_path = os.path.join(experiment_path, "model.pth")
    torch.save(model.state_dict(), model_path)

def test(dev_file, label_to_id, experiment_path):
    print("##### Test #####")

    test_vectors, test_labels = util.get_and_norm_test_data(dev_file, label_to_id, experiment_path)
    instances_count = test_vectors.shape[0]
    features_count = test_vectors.shape[1]
    labels_count = len(set(list(label_to_id.values())))

    print("number of instances: " + str(instances_count))

    print("test vectors shape: " + str(test_vectors.shape))
    print("test labels shape: " + str(test_labels.shape))

    test_labels = np.array(test_labels).reshape(-1, 1)

    tensor_test_x = torch.stack([torch.Tensor(i) for i in test_vectors])
    tensor_test_y = torch.stack([torch.Tensor(i) for i in test_labels])

    test_dataset = utils.TensorDataset(tensor_test_x, tensor_test_y)  # create your datset
    test_dataloader = utils.DataLoader(test_dataset, batch_size=32)  # create your dataloader

    model_path = os.path.join(experiment_path, "model.pth")
    model = Net(features_count, labels_count).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    predictions = []
    gold = []

    with torch.no_grad():
        for test_vectors, labels in test_dataloader:
            test_vectors = test_vectors.to(device)
            labels = labels.to(device, dtype=torch.int64).view(-1)

            outputs = model(test_vectors)
            _, predicted = torch.max(outputs.data, 1)

            m = nn.Softmax()
            s = m(outputs)

            predictions += predicted.data.tolist()
            gold += labels.data.tolist()

    print("#########################\n")
    print("UAR: " + str(recall_score(gold, predictions, average='macro')))
    print("Accuracy:" + str(accuracy(gold, predictions)))
    print("#########################\n")

    cm = ConfusionMatrix(gold, predictions)
    print(cm)

def eval_get_probabilities(dev_file, label_to_id, experiment_path):
    test_vectors, test_labels = util.get_and_norm_test_data(dev_file, label_to_id, experiment_path)

    features_count = test_vectors.shape[1]
    labels_count = len(set(list(label_to_id.values())))

    test_labels = np.array(test_labels).reshape(-1, 1)

    tensor_test_x = torch.stack([torch.Tensor(i) for i in test_vectors])
    tensor_test_y = torch.stack([torch.Tensor(i) for i in test_labels])

    test_dataset = utils.TensorDataset(tensor_test_x, tensor_test_y)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=32)

    model_path = os.path.join(experiment_path, "model.pth")
    model = Net(features_count, labels_count).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    probabilities = []
    softmax = nn.Softmax()

    with torch.no_grad():
        for test_vectors, _ in test_dataloader:
            test_vectors = test_vectors.to(device)

            outputs = model(test_vectors)
            _, predicted = torch.max(outputs.data, 1)

            probability = softmax(outputs).data.tolist()
            probabilities += probability

    return probabilities


#train(TRAIN_FILE_AUDIO, label_to_id, EXPERIMENT_PATH)
#test(DEV_FILE_AUDIO, label_to_id, EXPERIMENT_PATH)
#eval_get_probabilities(DEV_FILE_AUDIO, label_to_id, EXPERIMENT_PATH)