import numpy as np
import torch
import torch.utils.data as utils
from sklearn.svm import SVC
import torch.nn as nn
import torch.optim as optim
import pickle, os
from pytorchtools import EarlyStopping
from utils.experiments_util import get_metrics_str, sort_tensors, get_metrics

"""
This class is used everytime a pytorch model and a sklearn model are combined based on the output to train a new model.
Usually, one is the acoustic and one is the language model.
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

def index_dataset(dataset):
    indexed_dataset = {}
    for i in range(len(dataset.tensors[2])):
        features = dataset.tensors[0][i]
        label = dataset.tensors[1][i]
        id = dataset.tensors[2][i]
        id_as_int = int(id.data[0])
        indexed_dataset[id_as_int] = (features, label, id)
    return indexed_dataset

def get_dataset_instances_by_ids(indexed_dataset, target_ids):
    features_all = []
    labels = []
    ids = []

    for id in target_ids:
        features, label, id = indexed_dataset[int(id.data[0])]
        features_all.append(features)
        labels.append(label)
        ids.append(id)

    return torch.stack(features_all), torch.stack(labels), torch.stack(ids)

def train_mixed_modality_final_output_svm(pytorch_mod, sklearn_mod, id_to_name, experiment_path, logger, params):
    """
    Trains a Support Vector Machine based on the outputs of the two models and tests it
    :param pytorch_mod:
    :param sklearn_mod:
    :param id_to_name:
    :param experiment_path:
    :param model:
    :param logger:
    :param params:
    :return:
    """
    model1 = pytorch_mod['model'].to(device)
    model2 = sklearn_mod['model']

    train_instance_count1 = pytorch_mod['train_dataset'].tensors[0].size()[0]
    train_loader1 = utils.DataLoader(pytorch_mod['train_dataset'], shuffle=False, batch_size=train_instance_count1)
    index_train_dataset_sklearn = index_dataset(sklearn_mod['train_dataset'])

    test_instance_count1 = pytorch_mod['test_dataset'].tensors[0].size()[0]
    test_loader1 = utils.DataLoader(pytorch_mod['test_dataset'], shuffle=False, batch_size=test_instance_count1)
    index_test_dataset_sklearn = index_dataset(sklearn_mod['test_dataset'])

    h1 = model1.init_hidden(params["batch_size"])

    train_vectors = np.empty((0,8))
    train_labels = np.empty((0))

    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for inputs1, labels1, lengths1, ids1 in train_loader1:
            if inputs1.shape[0] != params["batch_size"]:
                continue
            lengths1, inputs1, labels1, ids1 = sort_tensors(lengths1, inputs1, labels1, ids1)
            inputs2, labels2, ids2 = get_dataset_instances_by_ids(index_train_dataset_sklearn, ids1)

            if not torch.all(torch.eq(ids1, ids2)):
                print('Expected the same instances for both modalities. Break')
                break

            labels1 = labels1.to(device, dtype=torch.int64).view(-1)
            lengths1 = lengths1.to(device, dtype=torch.int64).view(-1)

            h1 = tuple([each.data for each in h1])
            output1, _, _ = model1(inputs1.to(device), lengths1, h1)
            output1 = softmax(output1)

            output2 = np.array(model2.predict_proba(inputs2))

            added_outputs = np.append(output1.cpu().numpy(), output2, axis=1)
            train_vectors = np.concatenate((train_vectors, added_outputs), axis=0)
            train_labels = np.concatenate((train_labels, labels1.cpu().numpy()), axis=0)

    classifier = SVC(gamma='scale', decision_function_shape='ovo')
    classifier.fit(train_vectors, train_labels)
    model_path = os.path.join(experiment_path, 'svm_model.pkl')
    f = open(model_path, 'wb')
    pickle.dump(classifier, f)
    f.close()

    test_vectors = np.empty((0,8))
    test_labels = np.empty((0))
    test_ids = []
    with torch.no_grad():
        for inputs1, labels1, lengths1, ids1 in test_loader1:
            if inputs1.shape[0] != params["batch_size"]:
                continue

            lengths1, inputs1, labels1, ids1 = sort_tensors(lengths1, inputs1, labels1, ids1)
            inputs2, labels2, ids2 = get_dataset_instances_by_ids(index_test_dataset_sklearn, ids1)

            if not torch.all(torch.eq(ids1, ids2)):
                print('Expected the same instances for both modalities. Break')
                break

            labels1 = labels1.to(device, dtype=torch.int64).view(-1)
            ids = ids1.to(device, dtype=torch.int64).view(-1)
            lengths1 = lengths1.to(device, dtype=torch.int64).view(-1)

            h1 = tuple([each.data for each in h1])
            output1, _, _ = model1(inputs1.to(device), lengths1, h1)
            output1 = softmax(output1)

            output2 = np.array(model2.predict_proba(inputs2))

            added_outputs = np.append(output1.cpu().numpy(), output2, axis=1)

            test_vectors = np.concatenate((test_vectors, added_outputs), axis=0)
            test_labels = np.concatenate((test_labels, labels1.cpu().numpy()), axis=0)
            test_ids += ids.data.tolist()

    test_predictions = np.array(classifier.predict(test_vectors))
    test_predictions = [str(int(i)) for i in test_predictions]
    test_golds = [str(int(i)) for i in test_labels]

    metrics_str = get_metrics_str(test_golds, test_predictions)
    logger.info(metrics_str)

    model_path = os.path.join(experiment_path, 'svm_model.pkl')
    f = open(model_path, 'wb')
    pickle.dump(classifier, f)
    f.close()

    log_results = metrics_str + "\n\n"
    log_results += "Predicted\tGold\tName\n"

    for i in range(len(test_predictions)):
        log_results += str(test_predictions[i]) + "\t" + str(test_golds[i]) + "\t" + id_to_name[test_ids[i]] + "\n"
    log_results_path = os.path.join(experiment_path, "results.txt")
    with open(log_results_path, "w") as f:
        f.write(log_results)

    return test_golds, test_predictions