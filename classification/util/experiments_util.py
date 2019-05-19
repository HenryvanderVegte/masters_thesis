import os, datetime
import torch
import json
import logging
from sklearn.metrics import recall_score, precision_recall_fscore_support
from nltk.metrics import ConfusionMatrix, accuracy

def sort_by_length(inputs, labels, lengths):
    """
    Inputs three tensors with same size in dim 1, sorts them based on the third tensor
    :param inputs:
    :param labels:
    :param lengths:
    :return:
    """
    sorted_inputs = torch.stack([x for _, x in sorted(zip(lengths, inputs), key=lambda pair: pair[0], reverse=True)])
    sorted_labels = torch.stack([x for _, x in sorted(zip(lengths, labels), key=lambda pair: pair[0], reverse=True)])
    sorted_lengths = torch.stack(sorted(lengths, reverse=True))
    return sorted_inputs, sorted_labels, sorted_lengths

def create_experiment(experiments_dir, label_to_id, description, use_timestamp):
    time_as_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
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
                json.dumps(label_to_id) + "\n\n")

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

def log_metrics(labels, predictions, logger):
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

    logger.info(out)