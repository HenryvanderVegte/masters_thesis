import torch
import torch.utils.data as utils
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def create_sequence_dataset(feature_dict, label_dict, seq_length):
    """
    Loads sequence data into a TensorDataset,
    sequences longer than seq_length will be cut,
    sequences shorter than seq_length will be zero-padded
    :param feature_dict:
    :param label_dict:
    :param seq_length:
    :return:
    """
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

def create_sequence_dataset(feature_dict, label_dict, seq_length, pad_vector):
    """
    TODO: Implement method correctly
    :param feature_dict:
    :param label_dict:
    :param seq_length:
    :param pad_vector:
    :return:
    """
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