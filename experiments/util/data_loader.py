import numpy as np

def load_from_binary(label_file, feature_vector_bin_file):
    label_file = open(label_file, "r")
    label_file_lines = label_file.readlines()

    labels = []
    for line in label_file_lines:
        labels.append(line.split('\t')[1])

    feature_vectors = np.load(feature_vector_bin_file)

    return labels, feature_vectors


def load_from_binary(label_file, feature_vector_bin_file, label_to_id):
    label_file = open(label_file, "r")
    label_file_lines = label_file.readlines()

    labels = []
    label_lines = []

    for i in range(len(label_file_lines)):
        label = label_file_lines[i].split('\t')[1][:-1]
        if label in label_to_id:
            labels.append(int(label_to_id[label]))
            label_lines.append(i)

    feature_vectors = np.load(feature_vector_bin_file)

    return labels, feature_vectors[label_lines,:]

def load_dict_from_binary(label_file, feature_vector_bin_file, label_to_id):
    label_file = open(label_file, "r")
    label_file_lines = label_file.readlines()

    label_dict = {}

    for i in range(len(label_file_lines)):
        split = label_file_lines[i][:-1].split('\t')
        id = split[0]
        label = split[1]
        if label in label_to_id:
            label_dict[id] = int(label_to_id[label])
    feature_vectors_dict = np.load(feature_vector_bin_file).item()

    # remove feature vectors with a label that is unwanted (not in the label_to_id dict)
    ids_to_remove = []
    for id in feature_vectors_dict.keys():
        if id not in label_dict:
            ids_to_remove.append(id)
    for id in ids_to_remove:
        del feature_vectors_dict[id]

    return label_dict, feature_vectors_dict


def load_means_and_stddevs(means_path, stddevs_path):
    means = []
    means_file = open(means_path, "r")
    means_file_lines = means_file.readlines()
    for line in means_file_lines:
        means.append(float(line))
    means = np.array(means)

    stddevs = []
    stddevs_file = open(stddevs_path, "r")
    stddevs_file_lines = stddevs_file.readlines()
    for line in stddevs_file_lines:
        stddevs.append(float(line))
    stddevs = np.array(stddevs)
    return means, stddevs
