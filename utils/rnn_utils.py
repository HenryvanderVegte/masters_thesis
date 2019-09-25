import numpy as np
import torch
import torch.utils.data as utils
import torch.nn as nn
import torch.optim as optim
import os
from pytorchtools import EarlyStopping
from utils.experiments_util import get_metrics_str, sort_tensors, get_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(train_dataset, validation_dataset, test_dataset, id_to_name, experiment_path, model, logger, params):
    model = model.to(device)
    logger.info(str(params))
    logger.info(model)

    # Loss and optimizer
    unique, counts = np.unique(train_dataset.tensors[1], return_counts=True)
    count_dict = dict(zip(unique, counts))
    weights = 1 / np.array(list(count_dict.values()))
    weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss()#weight=weights)
    #optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    logger.info(optimizer)

    early_stopping = EarlyStopping(patience=4)
    logger.info(early_stopping)
    for e in range(params["epochs"]):
        train_loader = utils.DataLoader(train_dataset, shuffle=True, batch_size=params["batch_size"])
        h = model.init_hidden(params["batch_size"])

        train_losses = []
        for inputs, labels, lengths, _ in train_loader:
            if inputs.shape[0] != params["batch_size"]:
                continue
            lengths, inputs, labels = sort_tensors(lengths, inputs, labels)

            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.int64).view(-1)
            lengths = lengths.to(device, dtype=torch.int64).view(-1)
            h = tuple([each.data for each in h])

            model.zero_grad()
            output, h, _ = model(inputs, lengths, h)

            loss = criterion(output, labels)
            loss.backward()

            train_losses.append(loss.item())
            #nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        validation_losses = []
        validation_predictions = []
        validation_golds = []
        validation_instance_count = validation_dataset.tensors[0].size()[0]

        # batching to avoid running out of memory:
        if validation_instance_count > 10000:
            validation_instance_count = 256

        h = model.init_hidden(validation_instance_count)
        validation_loader = utils.DataLoader(validation_dataset, shuffle=True, batch_size=validation_instance_count)
        with torch.no_grad():
            for inputs, labels, lengths, _ in validation_loader:
                if inputs.shape[0] != validation_instance_count:
                    h = model.init_hidden(inputs.shape[0])
                lengths, inputs, labels = sort_tensors(lengths, inputs, labels)

                inputs = inputs.to(device)
                labels = labels.to(device, dtype=torch.int64).view(-1)
                lengths = lengths.to(device, dtype=torch.int64).view(-1)

                h = tuple([each.data for each in h])
                output, _, _ = model(inputs, lengths, h)

                validation_losses.append(criterion(output, labels).item())

                _, predicted = torch.max(output.data, 1)
                validation_predictions += predicted.data.tolist()
                validation_golds += labels.data.tolist()

        acc, _, _, _ = get_metrics(validation_golds, validation_predictions)
        logger.info('Epoch:{}/{:.0f}; Train loss:{:.4f}; Validation loss:{:.4f}; Validation accuracy:{:.4f}'.format(e, params["epochs"], np.mean(train_losses), np.mean(validation_losses), acc))
        metrics_str = get_metrics_str(validation_golds, validation_predictions)
        logger.info('Validation metrics: ' + metrics_str)

        early_stopping(np.mean(validation_losses), model)
        if early_stopping.early_stop:
            logger.info('Stop training. Take model from epoch ' + str(early_stopping.best_epoch))
            break
        if e > 20:
            break
    best_model = early_stopping.best_model

    model_path = os.path.join(experiment_path, "model.pth")
    torch.save(best_model.state_dict(), model_path)

    test_losses = []
    test_predictions = []
    test_golds = []
    test_ids = []

    test_instance_count = test_dataset.tensors[0].size()[0]

    # batching to avoid running out of memory:
    if test_instance_count > 10000:
        test_instance_count = 256

    h = best_model.init_hidden(test_instance_count)
    test_loader = utils.DataLoader(test_dataset, shuffle=False, batch_size=test_instance_count)
    with torch.no_grad():
        for inputs, labels, lengths, ids in test_loader:
            if inputs.shape[0] != test_instance_count:
                h = best_model.init_hidden(inputs.shape[0])
            lengths, inputs, labels, ids = sort_tensors(lengths, inputs, labels, ids)

            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.int64).view(-1)
            lengths = lengths.to(device, dtype=torch.int64).view(-1)
            ids = ids.to(device, dtype=torch.int64).view(-1)
            h = tuple([each.data for each in h])

            output, _, _ = best_model(inputs, lengths, h)

            test_losses.append(criterion(output, labels).item())

            _, predicted = torch.max(output.data, 1)
            test_predictions += predicted.data.tolist()
            test_golds += labels.data.tolist()
            test_ids += ids.data.tolist()

    metrics_str = get_metrics_str(test_golds, test_predictions)
    logger.info(metrics_str)



    log_results = metrics_str + "\n\n"
    log_results += "Predicted\tGold\tName\n"

    for i in range(len(test_predictions)):
        log_results += str(test_predictions[i]) + "\t" + str(test_golds[i]) + "\t" + id_to_name[test_ids[i]] + "\n"
    log_results_path = os.path.join(experiment_path, "results.txt")
    with open(log_results_path, "w") as f:
        f.write(log_results)

    return test_golds, test_predictions

def train_multilabel(train_dataset, validation_dataset, test_dataset, id_to_name, experiment_path, model, logger, params):
    model = model.to(device)
    logger.info(str(params))
    logger.info(model)

    # Loss and optimizer
    unique, counts = np.unique(train_dataset.tensors[1], return_counts=True)
    count_dict = dict(zip(unique, counts))
    weights = 1 / np.array(list(count_dict.values()))
    weights = torch.FloatTensor(weights).cuda()
    #nn.KLDivLoss
    criterion = nn.KLDivLoss()#(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    logger.info(optimizer)

    early_stopping = EarlyStopping(patience=3)
    logger.info(early_stopping)
    for e in range(params["epochs"]):
        train_loader = utils.DataLoader(train_dataset, shuffle=True, batch_size=params["batch_size"])
        h = model.init_hidden(params["batch_size"])

        train_losses = []
        for inputs, labels, all_labels, lengths, _ in train_loader:
            if inputs.shape[0] != params["batch_size"]:
                continue
            lengths, inputs, labels, all_labels = sort_tensors(lengths, inputs, labels, all_labels)

            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.int64).view(-1)
            all_labels = all_labels.to(device, dtype=torch.float32)
            #all_labels[all_labels > 0] = 1

            lengths = lengths.to(device, dtype=torch.int64).view(-1)
            h = tuple([each.data for each in h])

            model.zero_grad()
            output, h, _ = model(inputs, lengths, h)

            loss = criterion(output, all_labels)
            loss.backward()

            train_losses.append(loss.item())
            #nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        validation_losses = []
        validation_predictions = []
        validation_golds = []
        validation_instance_count = validation_dataset.tensors[0].size()[0]
        h = model.init_hidden(validation_instance_count)
        validation_loader = utils.DataLoader(validation_dataset, shuffle=True, batch_size=validation_instance_count)
        with torch.no_grad():
            for inputs, labels, all_labels, lengths, _ in validation_loader:
                lengths, inputs, labels, all_labels = sort_tensors(lengths, inputs, labels, all_labels)

                inputs = inputs.to(device)
                labels = labels.to(device, dtype=torch.int64).view(-1)
                all_labels = all_labels.to(device, dtype=torch.float32)
                lengths = lengths.to(device, dtype=torch.int64).view(-1)

                h = tuple([each.data for each in h])
                output, _, _ = model(inputs, lengths, h)

                validation_losses.append(criterion(output, all_labels).item())

                _, predicted = torch.max(output.data, 1)
                validation_predictions += predicted.data.tolist()
                validation_golds += labels.data.tolist()

        acc, _, _, _ = get_metrics(validation_golds, validation_predictions)
        logger.info('Epoch:{}/{:.0f}; Train loss:{:.4f}; Validation loss:{:.4f}; Validation accuracy:{:.4f}'.format(e, params["epochs"], np.mean(train_losses), np.mean(validation_losses), acc))

        early_stopping(np.mean(validation_losses), model)
        if early_stopping.early_stop:
            logger.info('Stop training. Take model from epoch ' + str(early_stopping.best_epoch))
            break
    best_model = early_stopping.best_model

    test_losses = []
    test_predictions = []
    test_golds = []
    test_ids = []

    test_instance_count = test_dataset.tensors[0].size()[0]
    h = best_model.init_hidden(test_instance_count)
    test_loader = utils.DataLoader(test_dataset, shuffle=False, batch_size=test_instance_count)
    with torch.no_grad():
        for inputs, labels, all_labels, lengths, ids in test_loader:
            lengths, inputs, labels, all_labels, ids = sort_tensors(lengths, inputs, labels, all_labels, ids)

            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.int64).view(-1)
            lengths = lengths.to(device, dtype=torch.int64).view(-1)
            all_labels = all_labels.to(device, dtype=torch.float32)
            ids = ids.to(device, dtype=torch.int64).view(-1)
            h = tuple([each.data for each in h])

            output, _, _ = best_model(inputs, lengths, h)

            test_losses.append(criterion(output, all_labels).item())

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