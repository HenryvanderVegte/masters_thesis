import os
from classification.util.global_vars import *

labels_folder = os.path.join(ROOT_FOLDER, "datasets//IEMOCAP//labels_full")

dicts = []

for r, d, f in os.walk(labels_folder):
    for file in f:
        file_path = os.path.join(r, file)
        with open(file_path, "r") as txt:
            print(file_path)
            line = txt.readline()

            while line:
                if line.startswith("["):
                    dict = {}
                    dict["id"] = line.split('\t')[1]
                    dict["majority_label"] = line.split('\t')[2]

                    line = txt.readline()
                    while line.startswith('C-'):
                        if line.startswith('C-F') or line.startswith('C-M'):
                            dict["self_label"] = line.split('\t')[1]
                        else:
                            dict[line.split('\t')[0]] = line.split('\t')[1]
                        line = txt.readline()
                    dicts.append(dict)
                line = txt.readline()

labels = {}

for dict in dicts:
    majority_label = dict["majority_label"]
    if majority_label not in labels:
        labels[majority_label] = 1
    else:
        labels[majority_label] = labels[majority_label] + 1

for label in labels.keys():
    print(label + " " + str(labels[label]))
