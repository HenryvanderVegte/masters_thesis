import os, datetime
import torch
import json
import logging
import numpy as np
from sklearn.metrics import recall_score, precision_recall_fscore_support
from nltk.metrics import ConfusionMatrix, accuracy

def read_tsv_metadata(dataset):
    """
    Reads given .tsv metadata file and returns every instance in a list of dictionaries
    :param dataset:
    :return: list of dictionaries, each representing one instance
    """
    d = open(dataset).read().splitlines()
    headers = d[0].split("\t")
    ret = []
    for row in d[1:]:
        row = row.split("\t")
        ret.append({headers[i]: row[i] for i in range(len(row))})
    return ret

def sort_tensors(sort_by, *args):
    """
    :param sort_by:
    :param args:
    :return:
    """
    results = []
    results.append(torch.stack(sorted(sort_by, reverse=True)))

    for arg in args:
        sorted_tensor = torch.stack([x for _, x in sorted(zip(sort_by, arg), key=lambda pair: pair[0], reverse=True)])
        results.append(sorted_tensor)

    return tuple(results)

def create_experiment(experiments_dir, class_groups, description, use_timestamp):
    time_as_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_dir = os.path.join(experiments_dir, time_as_string + "_" + description) if use_timestamp else os.path.join(experiments_dir, description)

    os.makedirs(experiment_dir)
    logfile = os.path.join(experiment_dir, 'logfile.log')

    logger = logging.getLogger("experiment_" + description)
    logger.setLevel('DEBUG')
    file_log_handler = logging.FileHandler(logfile)
    logger.addHandler(file_log_handler)

    stderr_log_handler = logging.StreamHandler()
    logger.addHandler(stderr_log_handler)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_log_handler.setFormatter(formatter)
    stderr_log_handler.setFormatter(formatter)

    logger.info("Experiment:\n" +
                description + "\n\n" +
                "Labels: \n" +
                json.dumps(class_groups) + "\n\n")

    return experiment_dir, logger

def get_ids_and_labels(feature_file, label_to_id):
    ids = []
    labels = []
    file = open(feature_file, "r")
    lines = file.readlines()
    for line in lines:
        split = line.split('\t')
        id = split[0]
        label = split[1]
        if label in label_to_id:
            ids.append(id)
            labels.append(label)
    return ids, labels

def join_ids_labels_probs(ids, labels, probs1, probs2):
    if not len(ids) == len(labels) == len(probs1) == len(probs2):
        print("Expected equal length will return")
        return
    out = ""
    for i in range(0, len(ids)):
        out += ids[i] + '\t' + labels[i] + "\t"
        for prob in probs1[i]:
            out += str(prob) + ","
        for prob in probs2[i]:
            out += str(prob) + ","
        out = out[:-1] + '\n'
    return out

def get_metrics_str(labels, predictions):
    out = "\n"

    cm = ConfusionMatrix(labels, predictions)
    out += "Confusion Matrix:\n" + str(cm) + "\n"

    out += "Accuracy:" + str(accuracy(labels, predictions)) + "\n"

    prec, rec, fscore, _ = precision_recall_fscore_support(labels, predictions)

    out += "Precision:\n"
    avg = 0
    for i, val in enumerate(prec):
        out += str(i) + " : " + str(val * 100) + " % \n"
        avg += val
    out += "Unweighted average precision:" + str((avg/len(prec) * 100)) + " % \n"

    out += "\nRecall:\n"
    avg = 0
    for i, val in enumerate(rec):
        out += str(i) + " : " + str(val * 100) + " % \n"
        avg += val
    out += "Unweighted average recall:" + str((avg/len(rec) * 100)) + " % \n"

    out += "\nF-measure:\n"
    avg = 0
    for i, val in enumerate(fscore):
        out += str(i) + " : " + str(val * 100) + " % \n"
        avg += val
    out += "Unweighted average f-measure:" + str((avg/len(fscore) * 100)) + " % \n"

    return out

def get_metrics(labels, predictions):
    acc = accuracy(labels, predictions)

    prec, rec, fscore, _ = precision_recall_fscore_support(labels, predictions)
    UAP = np.sum(prec) / len(prec)
    UAR = np.sum(rec) / len(rec)
    UAF = np.sum(fscore) / len(fscore)
    return acc, UAP, UAR, UAF

def get_means_and_stddevs_from_sequence_dataset(metadata, dataset_dict, class_groups, folds=None, take_gender = None):
    full_fl = []
    for instance in metadata:
        if instance["Label"] not in class_groups or instance['Name'] not in dataset_dict:
            continue

        if folds is not None and int(instance["Fold"]) not in folds:
            continue
        if take_gender is not None and instance['Gender'] != take_gender:
            continue

        for arr in dataset_dict[instance['Name']]:
            full_fl.append(arr)

    fl = np.array(full_fl)
    means = fl.mean(axis=0)
    stddevs = fl.std(axis=0)
    stddevs[stddevs == 0] = 1
    return means, stddevs

def get_means_and_stddevs_from_dataset(metadata, dataset_dict, class_groups, folds, take_gender = None):
    full_fl = []

    for instance in metadata:
        if instance["Label"] not in class_groups or int(instance["Fold"]) not in folds or instance['Name'] not in dataset_dict:
            continue

        if take_gender is not None and instance['Gender'] != take_gender:
            continue

        full_fl.append(dataset_dict[instance['Name']])

    fl = np.array(full_fl)
    means = fl.mean(axis=0)
    stddevs = fl.std(axis=0)
    stddevs[stddevs == 0] = 1

    return means, stddevs

def normalize_dataset(dataset_dict, means, stddevs):
    for key in dataset_dict.keys():
        dataset_dict[key] = (dataset_dict[key] - means) / stddevs
    return dataset_dict

def exclude_sequence_features_by_id(feature_dict, exclude_ids):
    for key in feature_dict.keys():
        new_seq = []
        for vec in feature_dict[key]:
            new_vec = np.delete(vec, exclude_ids)
            new_seq.append(new_vec)
        feature_dict[key] = np.array(new_seq)
        print(key)
    return feature_dict

def exclude_features_by_id(feature_dict, exclude_ids):
    for key in feature_dict.keys():
        new_fl = np.delete(feature_dict[key], exclude_ids)
        feature_dict[key] = np.asarray(new_fl)
    return feature_dict

def normalize_features(feature_dict):
    full_fl = []

    for fl in feature_dict.values():
        full_fl.append(fl)

    fl = np.array(full_fl)
    means = fl.mean(axis=0)
    stddevs = fl.std(axis=0)
    stddevs[stddevs == 0] = 1

    for key in feature_dict.keys():
        feature_dict[key] = (feature_dict[key] - means) / stddevs

    return feature_dict

def join_feature_dicts(feature_dict1, feature_dict2):
    joined_dict = {}
    keys = []
    for key in feature_dict1.keys():
        keys.append(key)

    for key in keys:
        joined_vecs = []

        if len(feature_dict1[key]) != len(feature_dict2[key]):
            print('Unequal lengths!')


        for i in range(len(feature_dict1[key])):
            vec1 = feature_dict1[key][i]
            vec2 = feature_dict2[key][i]
            joined_vecs.append(np.concatenate((vec1, vec2), axis=0))

        # delete old keys to save RAM:
        if key in feature_dict1:
            del feature_dict1[key]
        if key in feature_dict2:
            del feature_dict2[key]

        joined_dict[key] = np.asarray(joined_vecs)
    return joined_dict