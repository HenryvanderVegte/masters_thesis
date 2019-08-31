import torch
import os
import torch.nn as nn
import numpy as np
import torch.utils.data as utils
from classification.util import data_loader_pickle, data_loader_txt
from utils.experiments_util import get_metrics_str

TRAIN_FILE_AUDIO = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//split//train.txt"
DEV_FILE_AUDIO = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//split//dev.txt"
EXPERIMENT_PATH = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//classification//dnn"

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
num_epochs = 50
batch_size = 8
learning_rate = 0.01

class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1000)
        self.fc1_drop = nn.Dropout(p=0.3)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(1000, 100)
        self.fc2_drop = nn.Dropout(p=0.3)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(100, 20)
        self.fc3_drop = nn.Dropout(p=0.3)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(20, output_dim)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc1_drop(out)

        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc2_drop(out)

        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc3_drop(out)

        out = self.fc4(out)
        return out

def train(labels, features, normalize_features, experiment_path, logger):
    logger.info("############ Training DNN classifier. ########## \n\n" )
    if normalize_features:
        means = features.mean(axis=0)
        stddevs = features.std(axis=0)
        # remove 0 values
        stddevs[stddevs == 0] = 1
        features = (features - means) / stddevs

        means_path = os.path.join(experiment_path, 'means.txt')
        with open(means_path, "w") as f:
            for mean in means:
                f.write(str(mean) + '\n')

        logger.info("Saved means to " + means_path)

        stddevs_path = os.path.join(experiment_path, 'stddevs.txt')
        with open(stddevs_path, "w") as f:
            for stddev in stddevs:
                f.write(str(stddev) + '\n')
        logger.info("Saved stddevs to " + stddevs_path)

    labels_count = len(set(list(label_to_id.values())))
    instances_count = features.shape[0]
    features_count = features.shape[1]

    info = "labels: " + str(labels_count) + ", instances: " + str(instances_count) + ", features: " + str(features_count)
    logger.info("Data info: " + info)

    train_labels = np.array(labels).reshape(-1,1)

    tensor_train_x = torch.stack([torch.Tensor(i) for i in features])
    tensor_train_y = torch.stack([torch.Tensor(i) for i in train_labels])

    train_dataset = utils.TensorDataset(tensor_train_x,tensor_train_y) # create your datset
    train_dataloader = utils.DataLoader(train_dataset, batch_size=batch_size) # create your dataloader

    model = Net(features_count, labels_count).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)

    # Train the model
    total_step = len(train_dataloader)
    logger.info("Start train")
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
                logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    logger.info("Completed train")
    # export onnx model
    onnx_model_path = os.path.join(experiment_path, "dnn_text.onnx")
    dummy_input = torch.randn(batch_size, features_count, device='cuda')
    torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True)

    # export pytorch model
    model_path = os.path.join(experiment_path, "dnn_text.pth")
    torch.save(model.state_dict(), model_path)
    logger.info("Saved model to" + onnx_model_path + " and " + model_path)

def test(labels, features, normalize_features, experiment_path, logger):
    logger.info("############ Testing DNN classifier. ########## \n\n" )

    if normalize_features:
        means_path = os.path.join(experiment_path, 'means.txt')
        stddevs_path = os.path.join(experiment_path, 'stddevs.txt')

        means, stddevs = data_loader_txt.load_means_and_stddevs(means_path, stddevs_path)

        features = (np.array(features) - means) / stddevs

    instances_count = features.shape[0]
    features_count = features.shape[1]
    labels_count = len(set(list(label_to_id.values())))

    info = "labels: " + str(labels_count) + ", instances: " + str(instances_count) + ", features: " + str(features_count)
    logger.info("Data info: " + info)

    test_labels = np.array(labels).reshape(-1, 1)

    tensor_test_x = torch.stack([torch.Tensor(i) for i in features])
    tensor_test_y = torch.stack([torch.Tensor(i) for i in test_labels])

    test_dataset = utils.TensorDataset(tensor_test_x, tensor_test_y)  # create your datset
    test_dataloader = utils.DataLoader(test_dataset, batch_size=32)  # create your dataloader

    model_path = os.path.join(experiment_path, "dnn_text.pth")
    model = Net(features_count, labels_count).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    logger.info("Loaded model from" + model_path)

    predictions = []
    gold = []

    logger.info("Start test")
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

    logger.info(get_metrics_str(gold, predictions))

def eval_get_probabilities(test_file_in, experiment_path, label_to_id, logger):
    logger.info("############ Getting prob scores for DNN classifier. ########## \n\n" )

    test_vectors, test_labels = data_loader_pickle.get_and_norm_test_data(test_file_in, label_to_id, experiment_path)

    instances_count = test_vectors.shape[0]
    features_count = test_vectors.shape[1]
    labels_count = len(set(list(label_to_id.values())))
    test_labels = np.array(test_labels).reshape(-1, 1)

    info = "labels: " + str(labels_count) + ", instances: " + str(instances_count) + ", features: " + str(features_count)
    logger.info("Data info: " + info)

    tensor_test_x = torch.stack([torch.Tensor(i) for i in test_vectors])
    tensor_test_y = torch.stack([torch.Tensor(i) for i in test_labels])

    test_dataset = utils.TensorDataset(tensor_test_x, tensor_test_y)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=32)

    model_path = os.path.join(experiment_path, "dnn_audio.pth")
    model = Net(features_count, labels_count).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    logger.info("Loaded model from" + model_path)

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
