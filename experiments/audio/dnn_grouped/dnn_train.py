import torch
import torch.nn as nn
import numpy as np
from experiments.audio.util import util
import torch.utils.data as utils
from collections import Counter
from itertools import chain

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

train_vectors, train_labels = util.get_and_norm_train_data(TRAIN_FILE_AUDIO, label_to_id, EXPERIMENT_PATH)

labels_count = len(set(list(label_to_id.values())))
instances_count = train_vectors.shape[0]
features_count = train_vectors.shape[1]

print("number of labels: " + str(labels_count))
print("number of instances: " + str(instances_count))
print("number of features: " + str(features_count))

print("train vectors shape: " + str(train_vectors.shape))
print("train labels shape: " + str(train_labels.shape))


train_labels = np.array(train_labels).reshape(-1,1)
print(train_labels.shape)
#one_hot_encoding = np.zeros((instances_count, labels_count))
#one_hot_encoding[np.arange(instances_count), train_labels] = 1


tensor_x = torch.stack([torch.Tensor(i) for i in train_vectors]) # transform to torch tensors

tensor_y = torch.stack([torch.Tensor(i) for i in train_labels])

my_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
my_dataloader = utils.DataLoader(my_dataset, batch_size=32) # create your dataloader

print("loaded data")

# Hyper-parameters
num_epochs = 5

hidden_size = 1024
batch_size = 32
learning_rate = 0.01


class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_dim)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu2(out)

        out = self.fc3(out)
        return out


model = Net(features_count, labels_count).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)

# Train the model
total_step = len(my_dataloader)
print("start train")
for epoch in range(num_epochs):
    for i, (train_vectors, labels) in enumerate(my_dataloader):
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