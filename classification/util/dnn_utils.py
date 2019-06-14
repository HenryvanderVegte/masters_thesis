import numpy as np
import torch
import torch.utils.data as utils
import torch.nn as nn
import torch.optim as optim
import os
from classification.util.experiments_util import get_metrics_str, get_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(train_dataset, dev_dataset, experiment_path, model, logger, params):
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
    optimizer = optim.Adam(model.parameters())

    for e in range(params["epochs"]):
        train_losses = []
        train_preds = []
        train_golds = []
        train_ids = []

        for inputs, labels, ids in train_loader:
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

            _, predicted = torch.max(output.data, 1)
            train_preds += predicted.data.tolist()
            train_golds += labels.data.tolist()
            train_ids += ids.data.tolist()

        test_losses = []
        test_preds = []
        test_golds = []
        test_ids = []

        with torch.no_grad():
            for inputs, labels, ids in dev_loader:
                if inputs.shape[0] != params["batch_size"]:
                    continue
                inputs = inputs.to(device)
                labels = labels.to(device, dtype=torch.int64).view(-1)
                ids = ids.to(device, dtype=torch.int64).view(-1)

                output = model(inputs)

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
                log_results += str(test_preds[i]) + "\t" + str(test_golds[i]) + "\t" + str(test_ids[i]) + "\n"
            log_results_path = os.path.join(epoch_path, "results.txt")
            with open(log_results_path, "w") as f:
                f.write(log_results)


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

def eval_get_probabilities(test_file_in, experiment_path, label_to_id, logger):
    logger.info("############ Getting prob scores for DNN classifier. ########## \n\n" )

    test_vectors, test_labels = data_loader_txt.get_test_data(test_file_in, label_to_id, experiment_path, True, logger)

    instances_count = test_vectors.shape[0]
    features_count = test_vectors.shape[1]
    labels_count = len(set(list(label_to_id.values())))
    test_labels = np.array(test_labels).reshape(-1, 1)

    info = "labels: " + str(labels_count) + ", instances: " + str(instances_count) + ", features: " + str(features_count)
    logger.info("Data info: " + info)

    tensor_test_x = torch.stack([torch.Tensor(i) for i in test_vectors])
    tensor_test_y = torch.stack([torch.Tensor(i) for i in test_labels])

    test_dataset = utils.TensorDataset(tensor_test_x, tensor_test_y)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=32)

    model_path = os.path.join(experiment_path, "dnn_audio.pth")
    model = Net(features_count, labels_count).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    logger.info("Loaded model from" + model_path)

    probabilities = []
    softmax = nn.Softmax()

    with torch.no_grad():
        for test_vectors, _ in test_dataloader:
            test_vectors = test_vectors.to(device)

            outputs = model(test_vectors)

            probability = softmax(outputs).data.tolist()
            probabilities += probability

    return probabilities
