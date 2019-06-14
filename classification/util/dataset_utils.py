import os
import torch.utils.data as utils
import torch
import numpy as np
from enum import Enum

class Normalization(Enum):
    NO_NORM = 0,
    CREATE_NORM = 1,
    USE_NORM = 2

def create_emobase_dataset_from_metadata(metadata, class_groups, set, datasets_path, normalization, experiments_dir, datasets = None):
    fl = []
    labels = []
    ids = []

    for instance in metadata:
        if instance["Label"] not in class_groups or instance["Set"] != set:
            continue

        if datasets is not None and instance["Dataset"] not in datasets:
            continue

        label = class_groups[instance["Label"]]
        emobase_path = os.path.join(datasets_path, instance["Dataset"], "features//audio//emobase", instance['Name'] + '.emobase')

        with open(emobase_path, "r") as f:
            lines = f.readlines()
            emobase_features = [float(i) for i in lines]

        fl.append(emobase_features)

        labels.append(label)
        ids.append(int(instance["Id"]))

    fl = np.array(fl)

    if normalization != Normalization.NO_NORM:
        if normalization == Normalization.CREATE_NORM:
            means = fl.mean(axis=0)
            stddevs = fl.std(axis=0)
            stddevs[stddevs == 0] = 1

            means_path = os.path.join(experiments_dir, "means.txt")
            with open(means_path, "w") as f:
                for mean in means:
                    f.write(str(mean) + '\n')
            stddevs_path = os.path.join(experiments_dir, "stddevs.txt")
            with open(stddevs_path, "w") as f:
                for stddev in stddevs:
                    f.write(str(stddev) + '\n')
        elif normalization == Normalization.USE_NORM:
            means = []
            means_path = os.path.join(experiments_dir, "means.txt")
            with open(means_path, "r") as f:
                means_file_lines = f.readlines()
                for line in means_file_lines:
                    means.append(float(line))
            means = np.array(means)

            stddevs = []
            stddevs_path = os.path.join(experiments_dir, "stddevs.txt")
            with open(stddevs_path, "r") as f:
                stddevs_file_lines = f.readlines()
                for line in stddevs_file_lines:
                    stddevs.append(float(line))
                stddevs = np.array(stddevs)

        fl = (fl - means) / stddevs

    labels = np.array(labels).reshape(-1,1)
    ids = np.array(ids).reshape(-1,1)

    fl = torch.stack([torch.Tensor(i) for i in fl])
    labels = torch.stack([torch.Tensor(i) for i in labels])
    ids = torch.stack([torch.Tensor(i) for i in ids])

    dataset = utils.TensorDataset(fl, labels, ids)

    return dataset
