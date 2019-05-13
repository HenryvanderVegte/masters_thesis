import torch
import torch.utils.data as utils
from torch.nn.utils.rnn import pad_sequence
import numpy as np

'''
Loads sequence data into a TensorDataset,
sequences longer than seq_length will be cut, 
sequences shorter than seq_length will be padded
'''
def create_sequence_dataset(feature_dict, label_dict, seq_length):
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