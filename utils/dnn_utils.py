import numpy as np
import torch
import torch.utils.data as utils
import torch.nn as nn
import torch.optim as optim
import os
from pytorchtools import EarlyStopping
from utils.experiments_util import get_metrics_str, get_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)

def train(train_dataset, validation_dataset, test_dataset, id_to_name, experiment_path, model, logger, params):
    model = model.to(device)
    logger.info(str(params))
    logger.info(model)

    train_loader = utils.DataLoader(train_dataset, shuffle=True, batch_size=params["batch_size"])
    validation_loader = utils.DataLoader(validation_dataset, shuffle=False, batch_size=validation_dataset.tensors[0].size()[0])
    test_loader = utils.DataLoader(test_dataset, shuffle=False, batch_size=test_dataset.tensors[0].size()[0])

    # Loss and optimizer
    unique, counts = np.unique(train_dataset.tensors[1], return_counts=True)
    count_dict = dict(zip(unique, counts))
    weights = 1 / np.array(list(count_dict.values()))
    weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    #optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False)
    optimizer = optim.Adam(model.parameters())

    early_stopping = EarlyStopping(verbose=True, patience=20)
    for e in range(params["epochs"]):
        train_losses = []

        for inputs, labels, _ in train_loader:
            if inputs.shape[0] != params["batch_size"]:
                continue
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.int64).view(-1)

            model.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()

            train_losses.append(loss.item())
            optimizer.step()

        validation_losses = []
        validation_predictions = []
        validation_golds = []

        with torch.no_grad():
            for inputs, labels, _ in validation_loader:
                if inputs.shape[0] != params["batch_size"]:
                    continue
                inputs = inputs.to(device)
                labels = labels.to(device, dtype=torch.int64).view(-1)

                output = model(inputs)

                validation_losses.append(criterion(output, labels).item())

                _, predicted = torch.max(output.data, 1)
                validation_predictions += predicted.data.tolist()
                validation_golds += labels.data.tolist()

        acc, _, _, _ = get_metrics(validation_golds, validation_predictions)
        logger.info('Epoch:{}/{:.0f}; Train loss:{:.4f}; Validation loss:{:.4f}; Validation accuracy:{:.4f}'.format(e + 1, params["epochs"], np.mean(train_losses), np.mean(validation_losses), acc))

        early_stopping(np.mean(validation_losses), model)
        if early_stopping.early_stop:
            logger.info('Stop training. Take model from epoch ' + str(early_stopping.best_epoch))
            break

    best_model = early_stopping.best_model

    test_losses = []
    test_predictions = []
    test_golds = []
    test_ids = []
    with torch.no_grad():
        for inputs, labels, ids in test_loader:
            if inputs.shape[0] != params["batch_size"]:
                continue

            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.int64).view(-1)
            ids = ids.to(device, dtype=torch.int64).view(-1)

            output = best_model(inputs)

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

def test(dev_dataset, model, logger, params):
    dev_loader = utils.DataLoader(dev_dataset, shuffle=False, batch_size=params["batch_size"])

    predictions = []
    gold = []
    with torch.no_grad():
        for inputs, labels in dev_loader:
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.int64).view(-1)

            output = model(inputs)

            _, predicted = torch.max(output.data, 1)
            predictions += predicted.data.tolist()
            gold += labels.data.tolist()

    logger.info(get_metrics_str(gold, predictions))