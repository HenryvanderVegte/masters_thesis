import os, datetime
import json
import logging

def create_experiment(experiments_dir, label_to_id, description):
    time_as_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    experiment_dir = os.path.join(experiments_dir, time_as_string + "_" + description)

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