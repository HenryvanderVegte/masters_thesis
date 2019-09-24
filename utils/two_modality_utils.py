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
This class is used everytime two pretrained pytorch models are combined (early or late) based on the hidden layer
activation to get an output.
Usually, one is the acoustic and one is the language model.
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(17092019)

def index_dataset(dataset):
    indexed_dataset = {}
    for i in range(len(dataset.tensors[3])):
        features = dataset.tensors[0][i]
        label = dataset.tensors[1][i]
        length = dataset.tensors[2][i]
        id = dataset.tensors[3][i]

        id_as_int = int(id.data[0])
        indexed_dataset[id_as_int] = (features, label, length, id)
    return indexed_dataset

def get_dataset_instances_by_ids(indexed_dataset, target_ids):
    features_all = []
    labels = []
    lengths = []
    ids = []

    for id in target_ids:
        features, label, length, id = indexed_dataset[int(id.data[0])]
        features_all.append(features)
        labels.append(label)
        lengths.append(length)
        ids.append(id)

    return torch.stack(features_all), torch.stack(labels), torch.stack(lengths), torch.stack(ids)

def train_two_modality_rnn_join_hidden(resources_modality_1, resources_modality_2, id_to_name, experiment_path, joined_model, logger, params):
    """
    Trains a new RNN based on the hidden layer activation of two RNNs for each word (one acoustic RNN and one textual RNN)
    :param resources_modality_1:
    :param resources_modality_2:
    :param id_to_name:
    :param experiment_path:
    :param model:
    :param logger:
    :param params:
    :return:
    """
    joined_model = joined_model.to(device)
    logger.info(str(params))
    logger.info('Joined model: ' + str(joined_model))

    model1 = resources_modality_1['model'].to(device)
    model2 = resources_modality_2['model'].to(device)

    indexed_ds_train2 = index_dataset(resources_modality_2['train_dataset'])
    indexed_ds_validation2 = index_dataset(resources_modality_2['validation_dataset'])
    indexed_ds_test2 = index_dataset(resources_modality_2['test_dataset'])

    # Loss and optimizer
    unique, counts = np.unique(resources_modality_1['train_dataset'].tensors[1], return_counts=True)
    count_dict = dict(zip(unique, counts))
    weights = 1 / np.array(list(count_dict.values()))
    weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=weights)

    #optimizer = optim.Adam(joined_model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False)
    optimizer = optim.Adam(joined_model.parameters())
    logger.info(optimizer)

    early_stopping = EarlyStopping()
    logger.info(early_stopping)

    for e in range(params["epochs"]):
        train_loader1 = utils.DataLoader(resources_modality_1['train_dataset'], shuffle=True,
                                         batch_size=params["batch_size"])

        train_losses = []
        h = joined_model.init_hidden(params["batch_size"])
        h1 = model1.init_hidden(params["batch_size"])
        h2 = model2.init_hidden(params["batch_size"])

        for inputs1, labels1, lengths1, ids1 in train_loader1:
            if inputs1.shape[0] != params["batch_size"]:
                continue

            inputs2, labels2, lengths2, ids2 =get_dataset_instances_by_ids(indexed_ds_train2, ids1)

            if not torch.all(torch.eq(ids1, ids2)):
                print('Expected the same instances for both modalities. Break')
                break

            lengths1, inputs1, labels1, ids1 = sort_tensors(lengths1, inputs1, labels1, ids1)
            lengths2, inputs2, labels2, ids2 = sort_tensors(lengths2, inputs2, labels2, ids2)

            with torch.no_grad():
                h1 = tuple([each.data for each in h1])
                _, _, weights1 = model1(inputs1.to(device), lengths1.to(device, dtype=torch.int64).view(-1), h1)

                h2 = tuple([each.data for each in h2])
                _, _, weights2 = model2(inputs2.to(device), lengths2.to(device, dtype=torch.int64).view(-1), h2)

            #combine weights of the two other models to create the input for the joined model
            joined_inputs = torch.cat((weights1, weights2), 2)
            h = tuple([each.data for each in h])
            joined_model.zero_grad()
            output, h, _ = joined_model(joined_inputs, lengths1.to(device, dtype=torch.int64).view(-1), h)

            loss = criterion(output, labels1.to(device, dtype=torch.int64).view(-1))
            loss.backward()

            train_losses.append(loss.item())
            #nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        validation_losses = []
        validation_predictions = []
        validation_golds = []
        validation_instance_count = resources_modality_1['validation_dataset'].tensors[0].size()[0]
        validation_loader1 = utils.DataLoader(resources_modality_1['validation_dataset'], shuffle=False,
                                              batch_size=validation_instance_count)
        h = joined_model.init_hidden(validation_instance_count)
        h1 = model1.init_hidden(validation_instance_count)
        h2 = model2.init_hidden(validation_instance_count)

        with torch.no_grad():
            for inputs1, labels1, lengths1, ids1 in validation_loader1:
                inputs2, labels2, lengths2, ids2 = get_dataset_instances_by_ids(indexed_ds_validation2, ids1)
                if not torch.all(torch.eq(ids1, ids2)):
                    print('Expected the same instances for both modalities. Break')
                    break

                lengths1, inputs1, labels1 = sort_tensors(lengths1, inputs1, labels1)
                lengths2, inputs2, labels2 = sort_tensors(lengths2, inputs2, labels2)

                labels = labels1.to(device, dtype=torch.int64).view(-1)
                lengths = lengths1.to(device, dtype=torch.int64).view(-1)

                h1 = tuple([each.data for each in h1])
                _, _, weights1 = model1(inputs1.to(device), lengths, h1)

                h2 = tuple([each.data for each in h2])
                _, _, weights2 = model2(inputs2.to(device), lengths, h2)

                joined_inputs = torch.cat((weights1, weights2), 2)
                h = tuple([each.data for each in h])
                output, _, _ = joined_model(joined_inputs, lengths, h)
                validation_losses.append(criterion(output, labels).item())
                _, predicted = torch.max(output.data, 1)
                validation_predictions += predicted.data.tolist()
                validation_golds += labels.data.tolist()

        acc, _, _, _ = get_metrics(validation_golds, validation_predictions)
        logger.info('Epoch:{}/{:.0f}; Train loss:{:.4f}; Validation loss:{:.4f}; Validation accuracy:{:.4f}'.format(e + 1, params["epochs"], np.mean(train_losses), np.mean(validation_losses), acc))

        early_stopping(np.mean(validation_losses), joined_model)
        if early_stopping.early_stop:
            logger.info('Stop training. Take model from epoch ' + str(early_stopping.best_epoch))
            break

    best_model = early_stopping.best_model

    test_losses = []
    test_predictions = []
    test_golds = []
    test_ids = []
    test_instance_count = resources_modality_1['test_dataset'].tensors[0].size()[0]
    test_loader1 = utils.DataLoader(resources_modality_1['test_dataset'], shuffle=False,
                                    batch_size=test_instance_count)
    with torch.no_grad():
        h = best_model.init_hidden(test_instance_count)
        h1 = model1.init_hidden(test_instance_count)
        h2 = model2.init_hidden(test_instance_count)
        for inputs1, labels1, lengths1, ids1 in test_loader1:
            inputs2, labels2, lengths2, ids2 = get_dataset_instances_by_ids(indexed_ds_test2, ids1)
            if not torch.all(torch.eq(ids1, ids2)):
                print('Expected the same instances for both modalities. Break')
                break

            lengths, inputs1, labels, ids = sort_tensors(lengths1, inputs1, labels1, ids1)
            _, inputs2 = sort_tensors(lengths2, inputs2)

            labels = labels.to(device, dtype=torch.int64).view(-1)
            ids = ids.to(device, dtype=torch.int64).view(-1)
            lengths = lengths.to(device, dtype=torch.int64).view(-1)

            h1 = tuple([each.data for each in h1])
            _, _, weights1 = model1(inputs1.to(device), lengths, h1)

            h2 = tuple([each.data for each in h2])
            _, _, weights2 = model2(inputs2.to(device), lengths, h2)

            joined_inputs = torch.cat((weights1, weights2), 2)

            h = tuple([each.data for each in h])
            output, _, _ = best_model(joined_inputs, lengths, h)
            test_losses.append(criterion(output, labels).item())

            _, predicted = torch.max(output.data, 1)
            test_predictions += predicted.data.tolist()
            test_golds += labels.data.tolist()
            test_ids += ids.data.tolist()

    metrics_str = get_metrics_str(test_golds, test_predictions)
    logger.info(metrics_str)

    model_path = os.path.join(experiment_path, "model.pth")
    torch.save(best_model.state_dict(), model_path)

    log_results = metrics_str + "\n\n"
    log_results += "Predicted\tGold\tName\n"

    for i in range(len(test_predictions)):
        log_results += str(test_predictions[i]) + "\t" + str(test_golds[i]) + "\t" + id_to_name[test_ids[i]] + "\n"
    log_results_path = os.path.join(experiment_path, "results.txt")
    with open(log_results_path, "w") as f:
        f.write(log_results)

    return test_golds, test_predictions


def train_two_modality_max_prob_classifier(resources_modality_1, resources_modality_2, id_to_name, experiment_path, logger):
    """
    Selects the label with the highest probability for both modalities (text and audio) and classifies based on this
    :param resources_modality_1:
    :param resources_modality_2:
    :param logger:
    :return:
    """

    model1 = resources_modality_1['model'].to(device)
    model2 = resources_modality_2['model'].to(device)

    test_instance_count1 = resources_modality_1['test_dataset'].tensors[0].size()[0]
    test_loader1 = utils.DataLoader(resources_modality_1['test_dataset'], shuffle=False, batch_size=test_instance_count1)
    test_instance_count2 = resources_modality_2['test_dataset'].tensors[0].size()[0]
    test_loader2 = utils.DataLoader(resources_modality_2['test_dataset'], shuffle=False, batch_size=test_instance_count2)

    test_preds = []
    test_golds = []
    test_ids = []

    h1 = model1.init_hidden(test_instance_count1)
    h2 = model2.init_hidden(test_instance_count2)

    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        test_iter = iter(test_loader2)
        for inputs1, labels1, lengths1, ids1 in test_loader1:
            inputs2, labels2, lengths2, ids2 = next(test_iter)
            if not torch.all(torch.eq(ids1, ids2)):
                print('Expected the same instances for both modalities. Break')
                break

            lengths1, inputs1, labels1, ids1 = sort_tensors(lengths1, inputs1, labels1, ids1)
            lengths2, inputs2, labels2, ids2 = sort_tensors(lengths2, inputs2, labels2, ids2)

            labels1 = labels1.to(device, dtype=torch.int64).view(-1)
            ids1 = ids1.to(device, dtype=torch.int64).view(-1)
            lengths1 = lengths1.to(device, dtype=torch.int64).view(-1)
            lengths2 = lengths2.to(device, dtype=torch.int64).view(-1)

            h1 = tuple([each.data for each in h1])
            output1, _, _ = model1(inputs1.to(device), lengths1, h1)

            probability1 = softmax(output1)

            h2 = tuple([each.data for each in h2])
            output2, _, _ = model2(inputs2.to(device), lengths2, h2)
            probability2 = softmax(output2)

            added_probs = np.add(probability1.cpu().numpy(), probability2.cpu().numpy())

            predicted = np.argmax(added_probs, axis=1)
            test_preds += predicted.data.tolist()
            test_golds += labels1.data.tolist()
            test_ids += ids1.data.tolist()


    metrics_str = get_metrics_str(test_golds, test_preds)
    logger.info(metrics_str)

    log_results = metrics_str + "\n\n"
    log_results += "Predicted\tGold\tName\n"

    for i in range(len(test_preds)):
        log_results += str(test_preds[i]) + "\t" + str(test_golds[i]) + "\t" + id_to_name[test_ids[i]] + "\n"
    log_results_path = os.path.join(experiment_path, "results.txt")
    with open(log_results_path, "w") as f:
        f.write(log_results)

    return test_golds, test_preds

def train_two_modality_final_output_svm(resources_modality_1, resources_modality_2, id_to_name, experiment_path, logger):
    """
    Trains a Support Vector Machine based on the outputs of the two models and tests it
    :param resources_modality_1:
    :param resources_modality_2:
    :param id_to_name:
    :param experiment_path:
    :param model:
    :param logger:
    :return:
    """
    model1 = resources_modality_1['model'].to(device)
    model2 = resources_modality_2['model'].to(device)

    train_instance_count1 = 32#resources_modality_1['train_dataset'].tensors[0].size()[0]
    train_loader1 = utils.DataLoader(resources_modality_1['train_dataset'], shuffle=False, batch_size=train_instance_count1)
    train_instance_count2 = 32#resources_modality_2['train_dataset'].tensors[0].size()[0]
    train_loader2 = utils.DataLoader(resources_modality_2['train_dataset'], shuffle=False, batch_size=train_instance_count2)

    test_instance_count1 = resources_modality_1['test_dataset'].tensors[0].size()[0]
    test_loader1 = utils.DataLoader(resources_modality_1['test_dataset'], shuffle=False, batch_size=test_instance_count1)
    test_instance_count2 = resources_modality_2['test_dataset'].tensors[0].size()[0]
    test_loader2 = utils.DataLoader(resources_modality_2['test_dataset'], shuffle=False, batch_size=test_instance_count2)


    train_vectors = np.empty((0,6))
    train_labels = np.empty((0))

    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        train2_iter = iter(train_loader2)
        h1 = model1.init_hidden(train_instance_count1)
        h2 = model2.init_hidden(train_instance_count2)

        for inputs1, labels1, lengths1, ids1 in train_loader1:
            inputs2, labels2, lengths2, ids2 = next(train2_iter)
            if inputs1.shape[0] != train_instance_count1:
                continue

            if not torch.all(torch.eq(ids1, ids2)):
                print('Expected the same instances for both modalities. Break')
                break

            lengths1, inputs1, labels1, ids1 = sort_tensors(lengths1, inputs1, labels1, ids1)
            lengths2, inputs2, labels2, ids2 = sort_tensors(lengths2, inputs2, labels2, ids2)

            labels1 = labels1.to(device, dtype=torch.int64).view(-1)
            lengths1 = lengths1.to(device, dtype=torch.int64).view(-1)
            lengths2 = lengths2.to(device, dtype=torch.int64).view(-1)

            h1 = tuple([each.data for each in h1])
            output1, _, _ = model1(inputs1.to(device), lengths1, h1)
            output1 = softmax(output1)

            h2 = tuple([each.data for each in h2])
            output2, _, _ = model2(inputs2.to(device), lengths2, h2)
            output2 = softmax(output2)

            added_outputs = np.append(output1.cpu().numpy(), output2.cpu().numpy(), axis=1)

            train_vectors = np.concatenate((train_vectors, added_outputs), axis=0)
            train_labels = np.concatenate((train_labels, labels1.cpu().numpy()), axis=0)

    classifier = SVC(gamma='scale', decision_function_shape='ovo')
    classifier.fit(train_vectors, train_labels)
    model_path = os.path.join(experiment_path, 'svm_model.pkl')
    f = open(model_path, 'wb')
    pickle.dump(classifier, f)
    f.close()

    test_vectors = np.empty((0,6))
    test_labels = np.empty((0))
    test_ids = []
    with torch.no_grad():
        h1 = model1.init_hidden(test_instance_count1)
        h2 = model2.init_hidden(test_instance_count2)

        test2_iter = iter(test_loader2)
        for inputs1, labels1, lengths1, ids1 in test_loader1:
            inputs2, labels2, lengths2, ids2 = next(test2_iter)
            if not torch.all(torch.eq(ids1, ids2)):
                print('Expected the same instances for both modalities. Break')
                break

            lengths1, inputs1, labels1, ids1 = sort_tensors(lengths1, inputs1, labels1, ids1)
            lengths2, inputs2, labels2 = sort_tensors(lengths2, inputs2, labels2)

            labels1 = labels1.to(device, dtype=torch.int64).view(-1)
            ids = ids1.to(device, dtype=torch.int64).view(-1)
            lengths1 = lengths1.to(device, dtype=torch.int64).view(-1)
            lengths2 = lengths2.to(device, dtype=torch.int64).view(-1)

            h1 = tuple([each.data for each in h1])
            output1, _, _ = model1(inputs1.to(device), lengths1, h1)
            output1 = softmax(output1)

            h2 = tuple([each.data for each in h2])
            output2, _, _ = model2(inputs2.to(device), lengths2, h2)
            output2 = softmax(output2)

            added_outputs = np.append(output1.cpu().numpy(), output2.cpu().numpy(), axis=1)

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

def train_two_modality_final_activation_svm(resources_modality_1, resources_modality_2, id_to_name, experiment_path, logger, params):
    model1 = resources_modality_1['model'].to(device)
    model2 = resources_modality_2['model'].to(device)

    train_instance_count1 = 32#resources_modality_1['train_dataset'].tensors[0].size()[0]
    train_loader1 = utils.DataLoader(resources_modality_1['train_dataset'], shuffle=False, batch_size=train_instance_count1)
    train_instance_count2 = 32#resources_modality_2['train_dataset'].tensors[0].size()[0]
    train_loader2 = utils.DataLoader(resources_modality_2['train_dataset'], shuffle=False, batch_size=train_instance_count2)

    test_instance_count1 = resources_modality_1['test_dataset'].tensors[0].size()[0]
    test_loader1 = utils.DataLoader(resources_modality_1['test_dataset'], shuffle=False, batch_size=test_instance_count1)
    test_instance_count2 = resources_modality_2['test_dataset'].tensors[0].size()[0]
    test_loader2 = utils.DataLoader(resources_modality_2['test_dataset'], shuffle=False, batch_size=test_instance_count2)

    train_vectors = np.empty((0,params["activation_dims"]))
    train_labels = np.empty((0))

    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        train2_iter = iter(train_loader2)
        h1 = model1.init_hidden(train_instance_count1)
        h2 = model2.init_hidden(train_instance_count2)

        for inputs1, labels1, lengths1, ids1 in train_loader1:
            inputs2, labels2, lengths2, ids2 = next(train2_iter)
            if inputs1.shape[0] != train_instance_count1:
                continue

            if not torch.all(torch.eq(ids1, ids2)):
                print('Expected the same instances for both modalities. Break')
                break

            lengths1, inputs1, labels1, ids1 = sort_tensors(lengths1, inputs1, labels1, ids1)
            lengths2, inputs2, labels2, ids2 = sort_tensors(lengths2, inputs2, labels2, ids2)

            labels1 = labels1.to(device, dtype=torch.int64).view(-1)
            lengths1 = lengths1.to(device, dtype=torch.int64).view(-1)
            lengths2 = lengths2.to(device, dtype=torch.int64).view(-1)

            h1 = tuple([each.data for each in h1])
            _, _, activation1 = model1(inputs1.to(device), lengths1, h1)

            h2 = tuple([each.data for each in h2])
            _, _, activation2 = model2(inputs2.to(device), lengths2, h2)

            added_outputs = np.append(activation1.cpu().numpy(), activation2.cpu().numpy(), axis=1)

            train_vectors = np.concatenate((train_vectors, added_outputs), axis=0)
            train_labels = np.concatenate((train_labels, labels1.cpu().numpy()), axis=0)

    classifier = SVC(decision_function_shape='ovr')
    classifier.fit(train_vectors, train_labels)
    model_path = os.path.join(experiment_path, 'svm_model.pkl')
    f = open(model_path, 'wb')
    pickle.dump(classifier, f)
    f.close()

    test_vectors = np.empty((0,params["activation_dims"]))
    test_labels = np.empty((0))
    test_ids = []
    with torch.no_grad():
        h1 = model1.init_hidden(test_instance_count1)
        h2 = model2.init_hidden(test_instance_count2)

        test2_iter = iter(test_loader2)
        for inputs1, labels1, lengths1, ids1 in test_loader1:
            inputs2, labels2, lengths2, ids2 = next(test2_iter)
            if not torch.all(torch.eq(ids1, ids2)):
                print('Expected the same instances for both modalities. Break')
                break

            lengths1, inputs1, labels1, ids1 = sort_tensors(lengths1, inputs1, labels1, ids1)
            lengths2, inputs2, labels2 = sort_tensors(lengths2, inputs2, labels2)

            labels1 = labels1.to(device, dtype=torch.int64).view(-1)
            ids = ids1.to(device, dtype=torch.int64).view(-1)
            lengths1 = lengths1.to(device, dtype=torch.int64).view(-1)
            lengths2 = lengths2.to(device, dtype=torch.int64).view(-1)

            h1 = tuple([each.data for each in h1])
            _, _, activation1 = model1(inputs1.to(device), lengths1, h1)

            h2 = tuple([each.data for each in h2])
            _, _, activation2 = model2(inputs2.to(device), lengths2, h2)

            added_outputs = np.append(activation1.cpu().numpy(), activation2.cpu().numpy(), axis=1)

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

def train_two_modality_rnn_join_outputs(resources_modality_1, resources_modality_2, id_to_name, experiment_path, joined_model, logger, params):
    """
    Trains a new RNN based on the output of two RNNs for each word (one acoustic RNN and one textual RNN)
    :param resources_modality_1:
    :param resources_modality_2:
    :param id_to_name:
    :param experiment_path:
    :param model:
    :param logger:
    :param params:
    :return:
    """
    joined_model = joined_model.to(device)
    logger.info(str(params))
    logger.info('Joined model: ' + str(joined_model))

    model1 = resources_modality_1['model'].to(device)
    model2 = resources_modality_2['model'].to(device)

    indexed_ds_train2 = index_dataset(resources_modality_2['train_dataset'])
    indexed_ds_validation2 = index_dataset(resources_modality_2['validation_dataset'])
    indexed_ds_test2 = index_dataset(resources_modality_2['test_dataset'])

    # Loss and optimizer
    unique, counts = np.unique(resources_modality_1['train_dataset'].tensors[1], return_counts=True)
    count_dict = dict(zip(unique, counts))
    weights = 1 / np.array(list(count_dict.values()))
    weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=weights)

    #optimizer = optim.Adam(joined_model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False)
    optimizer = optim.Adadelta(joined_model.parameters(), lr=1e-2)
    logger.info(optimizer)

    early_stopping = EarlyStopping(patience=1)
    logger.info(early_stopping)
    #softmax = nn.Softmax(dim=2)

    for e in range(params["epochs"]):
        train_loader1 = utils.DataLoader(resources_modality_1['train_dataset'], shuffle=True,
                                         batch_size=params["batch_size"])
        train_losses = []
        h = joined_model.init_hidden(params["batch_size"])
        h1 = model1.init_hidden(params["batch_size"])
        h2 = model2.init_hidden(params["batch_size"])

        for inputs1, labels1, lengths1, ids1 in train_loader1:
            if inputs1.shape[0] != params["batch_size"]:
                continue

            inputs2, labels2, lengths2, ids2 =get_dataset_instances_by_ids(indexed_ds_train2, ids1)
            if not torch.all(torch.eq(ids1, ids2)):
                print('Expected the same instances for both modalities. Break')
                break

            lengths1, inputs1, labels1, ids1 = sort_tensors(lengths1, inputs1, labels1, ids1)
            lengths2, inputs2, labels2, ids2 = sort_tensors(lengths2, inputs2, labels2, ids2)

            with torch.no_grad():
                h1 = tuple([each.data for each in h1])
                _, _, _, outputs1 = model1(inputs1.to(device), lengths1.to(device, dtype=torch.int64).view(-1), h1)
                #outputs1 = softmax(outputs1)

                h2 = tuple([each.data for each in h2])
                _, _, _, outputs2 = model2(inputs2.to(device), lengths2.to(device, dtype=torch.int64).view(-1), h2)
                #outputs2 = softmax(outputs2)

            #combine weights of the two other models to create the input for the joined model
            joined_inputs = torch.cat((outputs1, outputs2), 2)
            h = tuple([each.data for each in h])
            joined_model.zero_grad()
            output, h, _ = joined_model(joined_inputs, lengths1.to(device, dtype=torch.int64).view(-1), h)

            loss = criterion(output, labels1.to(device, dtype=torch.int64).view(-1))
            loss.backward()

            train_losses.append(loss.item())
            #nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        h = joined_model.init_hidden(params["batch_size"])
        h1 = model1.init_hidden(params["batch_size"])
        h2 = model2.init_hidden(params["batch_size"])

        validation_losses = []
        validation_predictions = []
        validation_golds = []
        validation_loader1 = utils.DataLoader(resources_modality_1['validation_dataset'], shuffle=False,
                                              batch_size=params["batch_size"])
        with torch.no_grad():
            for inputs1, labels1, lengths1, ids1 in validation_loader1:
                if inputs1.shape[0] != params["batch_size"]:
                    continue

                inputs2, labels2, lengths2, ids2 = get_dataset_instances_by_ids(indexed_ds_validation2, ids1)
                if not torch.all(torch.eq(ids1, ids2)):
                    print('Expected the same instances for both modalities. Break')
                    break

                lengths1, inputs1, labels1 = sort_tensors(lengths1, inputs1, labels1)
                lengths2, inputs2, labels2 = sort_tensors(lengths2, inputs2, labels2)

                labels = labels1.to(device, dtype=torch.int64).view(-1)
                lengths = lengths1.to(device, dtype=torch.int64).view(-1)

                h1 = tuple([each.data for each in h1])
                _, _, _, outputs1 = model1(inputs1.to(device), lengths, h1)
                #outputs1 = softmax(outputs1)

                h2 = tuple([each.data for each in h2])
                _, _, _, outputs2 = model2(inputs2.to(device), lengths, h2)
                #outputs2 = softmax(outputs2)

                joined_inputs = torch.cat((outputs1, outputs2), 2)

                h = tuple([each.data for each in h])
                output, _, _ = joined_model(joined_inputs, lengths, h)
                validation_losses.append(criterion(output, labels).item())
                _, predicted = torch.max(output.data, 1)
                validation_predictions += predicted.data.tolist()
                validation_golds += labels.data.tolist()

        acc, _, _, _ = get_metrics(validation_golds, validation_predictions)
        logger.info('Epoch:{}/{:.0f}; Train loss:{:.4f}; Validation loss:{:.4f}; Validation accuracy:{:.4f}'.format(e, params["epochs"], np.mean(train_losses), np.mean(validation_losses), acc))

        early_stopping(np.mean(validation_losses), joined_model)
        #early_stopping(1-acc, joined_model)
        if early_stopping.early_stop:
            logger.info('Stop training. Take model from epoch ' + str(early_stopping.best_epoch))
            break

    best_model = early_stopping.best_model

    test_losses = []
    test_predictions = []
    test_golds = []
    test_ids = []

    test_instance_count = resources_modality_1['test_dataset'].tensors[0].size()[0]
    test_loader1 = utils.DataLoader(resources_modality_1['test_dataset'], shuffle=False, batch_size=test_instance_count)

    # batching to avoid running out of memory:
    if test_instance_count > 10000:
        test_instance_count = 256

    with torch.no_grad():
        h = best_model.init_hidden(test_instance_count)
        h1 = model1.init_hidden(test_instance_count)
        h2 = model2.init_hidden(test_instance_count)

        for inputs1, labels1, lengths1, ids1 in test_loader1:
            inputs2, labels2, lengths2, ids2 = get_dataset_instances_by_ids(indexed_ds_test2, ids1)

            if inputs1.shape[0] != test_instance_count:
                h = best_model.init_hidden(inputs1.shape[0])

            if not torch.all(torch.eq(ids1, ids2)):
                print('Expected the same instances for both modalities. Break')
                break

            lengths, inputs1, labels, ids = sort_tensors(lengths1, inputs1, labels1, ids1)
            _, inputs2 = sort_tensors(lengths2, inputs2)

            labels = labels.to(device, dtype=torch.int64).view(-1)
            ids = ids.to(device, dtype=torch.int64).view(-1)
            lengths = lengths.to(device, dtype=torch.int64).view(-1)

            h1 = tuple([each.data for each in h1])
            _, _, _, outputs1 = model1(inputs1.to(device), lengths, h1)
            #outputs1 = softmax(outputs1)

            h2 = tuple([each.data for each in h2])
            _, _, _, outputs2 = model2(inputs2.to(device), lengths, h2)
            #outputs2 = softmax(outputs2)

            joined_inputs = torch.cat((outputs1, outputs2), 2)

            h = tuple([each.data for each in h])
            output, _, _ = best_model(joined_inputs, lengths, h)
            test_losses.append(criterion(output, labels).item())

            _, predicted = torch.max(output.data, 1)
            test_predictions += predicted.data.tolist()
            test_golds += labels.data.tolist()
            test_ids += ids.data.tolist()

    metrics_str = get_metrics_str(test_golds, test_predictions)
    logger.info(metrics_str)

    model_path = os.path.join(experiment_path, "model.pth")
    torch.save(best_model.state_dict(), model_path)

    log_results = metrics_str + "\n\n"
    log_results += "Predicted\tGold\tName\n"

    for i in range(len(test_predictions)):
        log_results += str(test_predictions[i]) + "\t" + str(test_golds[i]) + "\t" + id_to_name[test_ids[i]] + "\n"
    log_results_path = os.path.join(experiment_path, "results.txt")
    with open(log_results_path, "w") as f:
        f.write(log_results)

    return test_golds, test_predictions