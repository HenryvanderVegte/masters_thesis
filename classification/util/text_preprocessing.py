import torch
import torch.utils.data as utils
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def create_sequence_dataset_from_dicts(feature_dict, label_dict, max_seq_length = None):
    """
    Loads sequence data into a TensorDataset,
    if max_seq_length is set, sequences longer than seq_length will be cut.
    sequences shorter than max_seq_length will be zero-padded
    :param feature_dict:
    :param label_dict:
    :param max_seq_length:
    :return:
    """
    fl = []
    labels = []
    lengths = []
    ids = []
    id_to_label = {}
    for i, key in enumerate(feature_dict.keys()):
        if max_seq_length is None:
            fl.append(torch.stack([torch.Tensor(i) for i in feature_dict[key]]))
        else:
            fl.append(torch.stack([torch.Tensor(i) for i in feature_dict[key][:max_seq_length, :]]))

        length = feature_dict[key].shape[0] if max_seq_length is None else min(feature_dict[key].shape[0], max_seq_length)
        lengths.append(length)
        labels.append(label_dict[key])
        ids.append(i)
        id_to_label[i] = key

    labels = np.array(labels).reshape(-1,1)
    lengths = np.array(lengths).reshape(-1,1)
    ids = np.array(ids).reshape(-1,1)
    padded_features = pad_sequence(fl)

    labels = torch.stack([torch.Tensor(i) for i in labels])
    lengths = torch.stack([torch.Tensor(i) for i in lengths])
    ids = torch.stack([torch.Tensor(i) for i in ids])

    padded_features = torch.transpose(padded_features, 0, 1)
    dataset = utils.TensorDataset(padded_features, labels, lengths, ids)

    return dataset, id_to_label

def create_sequence_dataset_with_pad_val(feature_dict, name_to_label_dict, seq_length, pad_val):
    """
    Creates a dataset of 1-dim vectors, cuts the vectors to :seq_length: and pads them with the :pad_val:
    :param feature_dict:
    :param name_to_label_dict:
    :param seq_length:
    :param pad_vector:
    :return: 1.) TensorDataset with features, labels, sequence lengths, and ids
            2.) Mapping for ids to names (i.e. 212: "Ses01F_impro01_F000")
    """
    features = []
    labels = []
    lengths = []
    ids = []
    id_to_name = {}
    for i, key in enumerate(feature_dict.keys()):
        cut_vec = feature_dict[key][:seq_length]
        pad_vec = np.full((seq_length), pad_val)
        pad_vec[:len(cut_vec)] = cut_vec

        lengths.append(min(feature_dict[key].shape[0], seq_length))
        features.append(torch.Tensor(pad_vec))
        labels.append(name_to_label_dict[key])
        ids.append(i)
        id_to_name[i] = key

    labels = np.array(labels).reshape(-1,1)
    lengths = np.array(lengths).reshape(-1,1)
    ids = np.array(ids).reshape(-1,1)

    labels = torch.stack([torch.Tensor(i) for i in labels])
    features = torch.stack([torch.Tensor(i) for i in features])
    lengths = torch.stack([torch.Tensor(i) for i in lengths])
    ids = torch.stack([torch.Tensor(i) for i in ids])

    dataset = utils.TensorDataset(features, labels, lengths, ids)

    return dataset, id_to_name
