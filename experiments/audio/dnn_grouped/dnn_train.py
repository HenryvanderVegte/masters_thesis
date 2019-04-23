import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as utils

TRAIN_FILE_AUDIO = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//split//train.txt"
MEANS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//experiments//dnn//means.txt"
STDDEVS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//experiments//dnn//stddevs.txt"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_classes = 10
num_epochs = 5

hidden_size = 1024
batch_size = 32
learning_rate = 0.01


my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])]
my_y = [np.array([4.]), np.array([2.])]




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


model = Net(12, 5).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.ptim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))