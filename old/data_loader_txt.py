import numpy as np
import os

def get_train_data(data_path, label_to_id, experiment_path, normalize_data, logger):
    logger.info("Load train data from " + str(data_path))
    logger.info("Normalize data: " + str(normalize_data))
    train_file = open(data_path, "r")
    train_file_lines = train_file.readlines()

    train_vectors = []
    train_labels = []

    for line in train_file_lines:
        line = line.replace('\n', '')
        split = line.split('\t')
        label = split[1]
        features = split[2]
        if label in label_to_id:
            train_labels.append(int(label_to_id[label]))
            features = features.split(',')
            features = [float(i) for i in features]
            train_vectors.append(np.array(features))

    train_vectors = np.array(train_vectors)
    train_labels = np.array(train_labels)

    if normalize_data:
        means = train_vectors.mean(axis=0)
        stddevs = train_vectors.std(axis=0)

        # remove 0 values
        stddevs[stddevs == 0] = 1

        train_vectors = (train_vectors - means) / stddevs

        means_path = os.path.join(experiment_path, "means.txt")
        with open(means_path, "w") as f:
            for mean in means:
                f.write(str(mean) + '\n')
        logger.info("Wrote means to: " + str(means_path))

        stddevs_path = os.path.join(experiment_path, "stddevs.txt")
        with open(stddevs_path, "w") as f:
            for stddev in stddevs:
                f.write(str(stddev) + '\n')
        logger.info("Wrote standard deviations to: " + str(stddevs_path))

    return train_vectors, train_labels

def get_test_data(data_path, label_to_id, experiment_path, normalize_data, logger):
    logger.info("Load test data from " + str(data_path))
    logger.info("Normalize data: " + str(normalize_data))
    test_file = open(data_path, "r")
    test_file_lines = test_file.readlines()

    test_vectors = []
    test_labels = []

    for line in test_file_lines:
        line = line.replace('\n', '')
        split = line.split('\t')
        label = split[1]
        features = split[2]
        if label in label_to_id:
            test_labels.append(int(label_to_id[label]))
            features = features.split(',')
            features = [float(i) for i in features]
            test_vectors.append(np.array(features))

    test_vectors = np.array(test_vectors)
    test_labels = np.array(test_labels)

    if normalize_data:
        means = []
        means_path = os.path.join(experiment_path, "means.txt")
        logger.info("Read means from: " + str(means_path))
        means_file = open(means_path, "r")
        means_file_lines = means_file.readlines()
        for line in means_file_lines:
            means.append(float(line))
        means = np.array(means)


        stddevs = []
        stddevs_path = os.path.join(experiment_path, "stddevs.txt")
        logger.info("Read standard deviations from: " + str(stddevs_path))
        stddevs_file = open(stddevs_path, "r")
        stddevs_file_lines = stddevs_file.readlines()
        for line in stddevs_file_lines:
            stddevs.append(float(line))
        stddevs = np.array(stddevs)
        test_vectors = (test_vectors - means) / stddevs

    return test_vectors, test_labels