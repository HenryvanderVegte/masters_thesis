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