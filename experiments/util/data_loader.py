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
