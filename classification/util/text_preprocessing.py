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

def create_sequence_dataset_with_pad_val(feature_dict, label_dict, seq_length, pad_val):
    """
    Creates a dataset of 1-dim vectors, cuts the vectors to :seq_length: and pads them with the :pad_val:
    :param feature_dict:
    :param label_dict:
    :param seq_length:
    :param pad_vector:
    :return:
    """
    features = []
    labels = []
    for key in feature_dict.keys():
        cut_vec = feature_dict[key][:seq_length]
        pad_vec = np.full((seq_length), pad_val)
        pad_vec[:len(cut_vec)] = cut_vec

        print(len(pad_vec))
        features.append(torch.Tensor(pad_vec))
        labels.append(label_dict[key])

    labels = np.array(labels).reshape(-1,1)

    labels = torch.stack([torch.Tensor(i) for i in labels])
    features = torch.stack([torch.Tensor(i) for i in features])

    dataset = utils.TensorDataset(features, labels)

    return dataset
