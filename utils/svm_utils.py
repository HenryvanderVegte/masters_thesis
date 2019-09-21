import numpy as np
from sklearn.svm import SVC
import pickle, os
from utils.experiments_util import get_metrics_str

def train(train_dataset, test_dataset, id_to_name, experiment_path, logger):
    train_features = np.array(train_dataset.tensors[0])
    train_labels = np.ravel(np.array(train_dataset.tensors[1]))

    test_features = np.array(test_dataset.tensors[0])
    test_labels = np.ravel(np.array(test_dataset.tensors[1]))
    test_ids = np.ravel(np.array(test_dataset.tensors[2]))

    classifier = SVC(gamma='scale', decision_function_shape='ovo')
    classifier.fit(train_features, train_labels)

    model_path = os.path.join(experiment_path, 'svm_model.pkl')

    f = open(model_path, 'wb')
    pickle.dump(classifier, f)
    f.close()

    test_predictions = np.array(classifier.predict(test_features))
    test_predictions = [str(int(i)) for i in test_predictions]
    test_labels = [str(int(i)) for i in test_labels]

    metrics_str = get_metrics_str(test_labels, test_predictions)
    logger.info(metrics_str)

    fold_results = metrics_str + "\n\n"
    fold_results += "Predicted\tGold\tName\n"

    for i in range(len(test_predictions)):
        fold_results += str(test_predictions[i]) + "\t" + str(test_labels[i]) + "\t" + id_to_name[test_ids[i]] + "\n"
    fold_results_path = os.path.join(experiment_path, "results.txt")
    with open(fold_results_path, "w") as f:
        f.write(fold_results)

    return test_labels, test_predictions