import numpy as np
import torch
import torch.utils.data as utils
import torch.nn as nn
import torch.optim as optim
import os
from classification.util.experiments_util import get_metrics, sort_tensors

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(train_dataset, dev_dataset, id_to_name, experiment_path, model, logger, params):
    model = model.to(device)
    logger.info(str(params))
    logger.info(model)

    train_loader = utils.DataLoader(train_dataset, shuffle=True, batch_size=params["batch_size"])
    dev_loader = utils.DataLoader(dev_dataset, shuffle=True, batch_size=len(dev_dataset))

    # Loss and optimizer
    unique, counts = np.unique(train_dataset.tensors[1], return_counts=True)
    count_dict = dict(zip(unique, counts))
    weights = 1 / np.array(list(count_dict.values()))
    weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters())

    for e in range(params["epochs"]):
        h = model.init_hidden(params["batch_size"])
        train_losses = []
        for inputs, labels, lengths, _ in train_loader:
            lengths, inputs, labels = sort_tensors(lengths, inputs, labels)

            if inputs.shape[0] != params["batch_size"]:
                continue
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.int64).view(-1)
            lengths = lengths.to(device, dtype=torch.int64).view(-1)
            h = tuple([each.data for each in h])

            model.zero_grad()

            output, h = model(inputs, lengths, h)

            loss = criterion(output, labels)
            loss.backward()

            train_losses.append(loss.item())
            nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

        test_loss = 0
        all_predictions = []
        all_gold = []
        all_ids = []
        h = model.init_hidden(len(dev_dataset))
        with torch.no_grad():
            for inputs, labels, lengths, ids in dev_loader:
                lengths, inputs, labels, ids = sort_tensors(lengths, inputs, labels, ids)

                inputs = inputs.to(device)
                labels = labels.to(device, dtype=torch.int64).view(-1)
                lengths = lengths.to(device, dtype=torch.int64).view(-1)
                ids = ids.to(device, dtype=torch.int64).view(-1)
                h = tuple([each.data for each in h])

                output, _ = model(inputs, lengths, h)

                test_loss = criterion(output, labels)

                _, predicted = torch.max(output.data, 1)
                all_predictions += predicted.data.tolist()
                all_gold += labels.data.tolist()
                all_ids += ids.data.tolist()

        logger.info('{}; {:.0f} {:.4f}; {:.4f}'.format(e + 1, params["epochs"], np.mean(train_losses), test_loss))


        if (e+1) % int(params["log_x_epochs"]) == 0:
            logger.info("Epoch nr " + str(e))
            metrics_str = get_metrics(all_gold, all_predictions)
            logger.info(metrics_str)
            epoch_path = os.path.join(experiment_path, "epoch_" + str(e))
            os.mkdir(epoch_path)
            model_path = os.path.join(epoch_path,  "rnn.pth")
            torch.save(model.state_dict(), model_path)

            log_results = metrics_str +"\n\n"
            log_results += "Predicted\tGold\tName\n"
            for i in range(len(all_predictions)):
                log_results += str(all_predictions[i]) + "\t" + str(all_gold[i]) + "\t" + id_to_name[all_ids[i]] + "\n"
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

    logger.info(get_metrics(gold, predictions))