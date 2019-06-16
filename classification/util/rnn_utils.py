import numpy as np
import torch
import torch.utils.data as utils
import torch.nn as nn
import torch.optim as optim
import os
from classification.util.experiments_util import get_metrics_str, sort_tensors, get_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(train_dataset, dev_dataset, id_to_name, experiment_path, model, logger, params):
    log_epochs = os.path.join(experiment_path, "log_epochs.txt")
    with open(log_epochs, "w") as f:
        f.write('epochs;train_loss;train_acc;train_UAP;train_UAR;train_UAF;dev_loss;dev_acc;dev_UAP;dev_UAR;dev_UAF\n')

    model = model.to(device)
    logger.info(str(params))
    logger.info(model)

    train_loader = utils.DataLoader(train_dataset, shuffle=True, batch_size=params["batch_size"])
    dev_loader = utils.DataLoader(dev_dataset, shuffle=True, batch_size=params["batch_size"])

    # Loss and optimizer
    unique, counts = np.unique(train_dataset.tensors[1], return_counts=True)
    count_dict = dict(zip(unique, counts))
    weights = 1 / np.array(list(count_dict.values()))
    weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    for e in range(params["epochs"]):
        h = model.init_hidden(params["batch_size"])

        train_losses = []
        train_preds = []
        train_golds = []
        train_ids = []

        for inputs, labels, lengths, ids in train_loader:
            if inputs.shape[0] != params["batch_size"]:
                continue
            lengths, inputs, labels, ids = sort_tensors(lengths, inputs, labels, ids)


            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.int64).view(-1)
            lengths = lengths.to(device, dtype=torch.int64).view(-1)
            h = tuple([each.data for each in h])

            model.zero_grad()

            output, h = model(inputs, lengths, h)

            loss = criterion(output, labels)
            loss.backward()

            train_losses.append(loss.item())
            #nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

            _, predicted = torch.max(output.data, 1)
            train_preds += predicted.data.tolist()
            train_golds += labels.data.tolist()
            train_ids += ids.data.tolist()

        test_losses = []
        test_preds = []
        test_golds = []
        test_ids = []
        h = model.init_hidden(params["batch_size"])
        with torch.no_grad():
            for inputs, labels, lengths, ids in dev_loader:
                if inputs.shape[0] != params["batch_size"]:
                    continue
                lengths, inputs, labels, ids = sort_tensors(lengths, inputs, labels, ids)

                inputs = inputs.to(device)
                labels = labels.to(device, dtype=torch.int64).view(-1)
                lengths = lengths.to(device, dtype=torch.int64).view(-1)
                ids = ids.to(device, dtype=torch.int64).view(-1)
                h = tuple([each.data for each in h])

                output, _ = model(inputs, lengths, h)

                test_losses.append(criterion(output, labels).item())

                _, predicted = torch.max(output.data, 1)
                test_preds += predicted.data.tolist()
                test_golds += labels.data.tolist()
                test_ids += ids.data.tolist()

        logger.info('{}; {:.0f} {:.4f}; {:.4f}'.format(e + 1, params["epochs"], np.mean(train_losses), np.mean(test_losses)))

        epoch_metrics = str(e) + ';'

        acc, uap, uar, uaf = get_metrics(train_golds, train_preds)
        epoch_metrics += '{};{};{};{};{};'.format(np.mean(train_losses), acc, uap, uar, uaf)

        acc, uap, uar, uaf = get_metrics(test_golds, test_preds)
        epoch_metrics += '{};{};{};{};{}'.format(np.mean(test_losses), acc, uap, uar, uaf)

        with open(log_epochs, "a") as f:
            f.write(epoch_metrics + '\n')

        if (e+1) % int(params["log_x_epochs"]) == 0:
            logger.info("Epoch nr " + str(e))
            metrics_str = get_metrics_str(test_golds, test_preds)
            logger.info(metrics_str)
            epoch_path = os.path.join(experiment_path, "epoch_" + str(e))
            os.mkdir(epoch_path)
            model_path = os.path.join(epoch_path,  "rnn.pth")
            torch.save(model.state_dict(), model_path)

            log_results = metrics_str +"\n\n"
            log_results += "Predicted\tGold\tName\n"
            for i in range(len(test_preds)):
                log_results += str(test_preds[i]) + "\t" + str(test_golds[i]) + "\t" + id_to_name[test_ids[i]] + "\n"
            log_results_path = os.path.join(epoch_path, "results.txt")
            with open(log_results_path, "w") as f:
                f.write(log_results)


def test(dev_dataset, model, label_to_id, logger, params):
    dev_loader = utils.DataLoader(dev_dataset, shuffle=False, batch_size=params["batch_size"])

    predictions = []
    gold = []
    with torch.no_grad():
        for inputs, labels in dev_loader:
            inputs = inputs.to(device)
            #inputs = torch.transpose(inputs, 0, 1)
            labels = labels.to(device, dtype=torch.int64).view(-1)

            output = model(inputs)

            _, predicted = torch.max(output.data, 1)
            predictions += predicted.data.tolist()
            gold += labels.data.tolist()

    logger.info(get_metrics_str(gold, predictions))