import numpy as np

def get_and_norm_train_data(data_path, label_to_id, experiment_path):
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
            train_vectors.append(np.array(features.split(',')).astype(np.float))

    train_vectors = np.array(train_vectors)

    means = train_vectors.mean(axis=0)
    stddevs = train_vectors.std(axis=0)

    # remove 0 values
    stddevs[stddevs == 0] = 1

    normed_vectors = (train_vectors - means) / stddevs

    with open(means_out, "w") as f:
        for mean in means:
            f.write(str(mean) + '\n')

    with open(stddevs_out, "w") as f:
        for stddev in stddevs:
            f.write(str(stddev) + '\n')

    return normed_vectors