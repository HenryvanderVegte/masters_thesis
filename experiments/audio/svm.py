import numpy as np
from sklearn.svm import SVC
import pickle, os
from sklearn.metrics import recall_score
from nltk.metrics import ConfusionMatrix, accuracy

def __load_means_and_stddevs(means_path, stddevs_path):
    means = []
    means_file = open(means_path, "r")
    means_file_lines = means_file.readlines()
    for line in means_file_lines:
        means.append(float(line))
    means = np.array(means)

    stddevs = []
    stddevs_file = open(stddevs_path, "r")
    stddevs_file_lines = stddevs_file.readlines()
    for line in stddevs_file_lines:
        stddevs.append(float(line))
    stddevs = np.array(stddevs)
    return means, stddevs

def __get_labels_and_features(feature_file, label_to_id):
    labels = []
    feature_vectors = []

    feature_file = open(feature_file, "r")
    feature_file_lines = feature_file.readlines()
    for line in feature_file_lines:
        line = line.replace('\n', '')
        split = line.split('\t')
        label = split[1]
        features = split[2]
        if label in label_to_id:
            labels.append(int(label_to_id[label]))
            feature_vectors.append(np.array(features.split(',')).astype(np.float))

    feature_vectors = np.array(feature_vectors)
    return labels, feature_vectors


def train(train_file_in, experiment_dir, label_to_id, logger):
    logger.info("Training SVM classifier on " + str(train_file_in))

    labels, features = __get_labels_and_features(train_file_in, label_to_id)

    means = features.mean(axis=0)
    stddevs = features.std(axis=0)

    # remove 0 values
    stddevs[stddevs == 0] = 1

    normed_features = (features - means) / stddevs

    classifier = SVC(gamma='scale', decision_function_shape='ovo')
    classifier.fit(normed_features, labels)

    model_path = os.path.join(experiment_dir, 'svm_model.pkl')
    f = open(model_path, 'wb')
    pickle.dump(classifier, f)
    f.close()

    logger.info("Completed training. Saved model to " + model_path)

    means_path = os.path.join(experiment_dir, 'means.txt')
    with open(means_path, "w") as f:
        for mean in means:
            f.write(str(mean) + '\n')

    logger.info("Saved means to " + means_path)

    stddevs_path = os.path.join(experiment_dir, 'stddevs.txt')
    with open(stddevs_path, "w") as f:
        for stddev in stddevs:
            f.write(str(stddev) + '\n')
    logger.info("Saved stddevs to " + stddevs_path)

def test(test_file_in, experiment_dir, label_to_id, logger):
    logger.info("Training SVM classifier on " + str(test_file_in))

    means_path = os.path.join(experiment_dir, 'means.txt')
    stddevs_path = os.path.join(experiment_dir, 'stddevs.txt')
    means, stddevs = __load_means_and_stddevs(means_path, stddevs_path)

    labels, features = __get_labels_and_features(test_file_in, label_to_id)

    normed_features = (np.array(features) - means) / stddevs

    model_path = os.path.join(experiment_dir, 'svm_model.pkl')
    logger.info("Load model from " + str(model_path))
    f = open(model_path, 'rb')
    classifier = pickle.load(f)
    f.close()

    pred = np.array(classifier.predict(normed_features))
    pred = [str(i) for i in pred]

    test_labels = np.array(labels)
    test_labels = [str(i) for i in test_labels]

    logger.info("Accuracy:" + str(accuracy(test_labels, pred)))

    logger.info("Unweighted average recall:" + str(recall_score(test_labels, pred, average='macro')))

    cm = ConfusionMatrix(test_labels, pred)

    logger.info("Confusion Matrix:" + str(cm))
