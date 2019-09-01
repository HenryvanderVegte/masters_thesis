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
This class is used everytime two pretrained models are combined (early or late) to get an output.
Usually, one is the acoustic and one is the language model.
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

def train_two_modality_rnn(resources_modality_1, resources_modality_2, id_to_name, experiment_path, joined_model, logger, params):
    """
    Trains a new RNN based on the outputs of two RNNs for each word (one acoustic RNN and one textual RNN)
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

    train_loader1 = utils.DataLoader(resources_modality_1['train_dataset'], shuffle=False, batch_size=params["batch_size"])
    train_loader2 = utils.DataLoader(resources_modality_2['train_dataset'], shuffle=False, batch_size=params["batch_size"])

    validation_loader1 = utils.DataLoader(resources_modality_1['validation_dataset'], shuffle=False, batch_size=params["batch_size"])
    validation_loader2 = utils.DataLoader(resources_modality_2['validation_dataset'], shuffle=False, batch_size=params["batch_size"])

    test_loader1 = utils.DataLoader(resources_modality_1['test_dataset'], shuffle=False, batch_size=params["batch_size"])
    test_loader2 = utils.DataLoader(resources_modality_2['test_dataset'], shuffle=False, batch_size=params["batch_size"])

    # Loss and optimizer
    unique, counts = np.unique(resources_modality_1['train_dataset'].tensors[1], return_counts=True)
    count_dict = dict(zip(unique, counts))
    weights = 1 / np.array(list(count_dict.values()))
    weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(joined_model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False)
    #optimizer = optim.Adam(joined_model.parameters())

    early_stopping = EarlyStopping(verbose=True)

    for e in range(params["epochs"]):
        train_losses = []

        h = joined_model.init_hidden(params["batch_size"])
        h1 = model1.init_hidden(params["batch_size"])
        h2 = model2.init_hidden(params["batch_size"])

        train2_iter = iter(train_loader2)
        for inputs1, labels1, lengths1, ids1 in train_loader1:
            joined_model.train()
            if inputs1.shape[0] != params["batch_size"]:
                continue
            inputs2, labels2, lengths2, ids2 = next(train2_iter)

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

        h = joined_model.init_hidden(params["batch_size"])
        h1 = model1.init_hidden(params["batch_size"])
        h2 = model2.init_hidden(params["batch_size"])

        validation_losses = []
        validation_predictions = []
        validation_golds = []
        with torch.no_grad():
            joined_model.eval()
            validation2_iter = iter(validation_loader2)
            for inputs1, labels1, lengths1, ids1 in validation_loader1:
                if inputs1.shape[0] != params["batch_size"]:
                    continue

                inputs2, labels2, lengths2, ids2 = next(validation2_iter)
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
            print("Stopping training!")
            break
        print("Stopping training!")
        break

    best_model = early_stopping.best_model

    test_losses = []
    test_predictions = []
    test_golds = []
    test_ids = []
    with torch.no_grad():
        h = best_model.init_hidden(params["batch_size"])
        h1 = model1.init_hidden(params["batch_size"])
        h2 = model2.init_hidden(params["batch_size"])

        test2_iter = iter(validation_loader2)
        for inputs1, labels1, lengths1, ids1 in validation_loader1:
            if inputs1.shape[0] != params["batch_size"]:
                continue

            inputs2, labels2, lengths2, ids2 = next(test2_iter)
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


def train_two_modality_max_prob_classifier(resources_modality_1, resources_modality_2, logger, params):
    """
    Selects the label with the highest probability for both modalities (text and audio) and classifies based on this
    :param resources_modality_1:
    :param resources_modality_2:
    :param logger:
    :param params:
    :return:
    """

    model1 = resources_modality_1['model'].to(device)
    model2 = resources_modality_2['model'].to(device)

    dev_loader1 = utils.DataLoader(resources_modality_1['dev_dataset'], shuffle=False, batch_size=params["batch_size"])
    dev_loader2 = utils.DataLoader(resources_modality_2['dev_dataset'], shuffle=False, batch_size=params["batch_size"])

    test_preds = []
    test_golds = []
    test_ids = []

    h1 = model1.init_hidden(params["batch_size"])
    h2 = model2.init_hidden(params["batch_size"])

    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        dev2_iter = iter(dev_loader2)
        for inputs1, labels1, lengths1, ids1 in dev_loader1:
            if inputs1.shape[0] != params["batch_size"]:
                continue

            inputs2, labels2, lengths2, ids2 = next(dev2_iter)
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

    logger.info(get_metrics_str(test_golds, test_preds))

def train_two_modality_final_output_svm(resources_modality_1, resources_modality_2, id_to_name, experiment_path, logger, params):
    """
    Trains a Support Vector Machine based on the outputs of the two models and tests it
    :param resources_modality_1:
    :param resources_modality_2:
    :param id_to_name:
    :param experiment_path:
    :param model:
    :param logger:
    :param params:
    :return:
    """
    model1 = resources_modality_1['model'].to(device)
    model2 = resources_modality_2['model'].to(device)

    train_loader1 = utils.DataLoader(resources_modality_1['train_dataset'], shuffle=False, batch_size=params["batch_size"])
    train_loader2 = utils.DataLoader(resources_modality_2['train_dataset'], shuffle=False, batch_size=params["batch_size"])

    test_loader1 = utils.DataLoader(resources_modality_1['test_dataset'], shuffle=False, batch_size=params["batch_size"])
    test_loader2 = utils.DataLoader(resources_modality_2['test_dataset'], shuffle=False, batch_size=params["batch_size"])

    h1 = model1.init_hidden(params["batch_size"])
    h2 = model2.init_hidden(params["batch_size"])

    train_vectors = np.empty((0,8))
    train_labels = np.empty((0))

    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        train2_iter = iter(train_loader2)
        for inputs1, labels1, lengths1, ids1 in train_loader1:
            if inputs1.shape[0] != params["batch_size"]:
                continue

            inputs2, labels2, lengths2, ids2 = next(train2_iter)
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

    test_vectors = np.empty((0,8))
    test_labels = np.empty((0))
    test_ids = []
    with torch.no_grad():
        dev2_iter = iter(test_loader2)
        for inputs1, labels1, lengths1, ids1 in test_loader1:
            if inputs1.shape[0] != params["batch_size"]:
                continue

            inputs2, labels2, lengths2, ids2 = next(dev2_iter)
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