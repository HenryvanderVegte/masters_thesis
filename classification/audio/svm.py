import numpy as np
from sklearn.svm import SVC
import pickle, os
from sklearn.metrics import recall_score
from nltk.metrics import ConfusionMatrix, accuracy

def train(labels, feature_vectors, experiment_dir, logger):
    logger.info("Training SVM classifier." )
    classifier = SVC(gamma='scale', decision_function_shape='ovo')
    classifier.fit(feature_vectors, labels)

    model_path = os.path.join(experiment_dir, 'svm_model.pkl')
    f = open(model_path, 'wb')
    pickle.dump(classifier, f)
    f.close()

    logger.info("Completed training. Saved model to " + model_path)

def test(labels, feature_vectors, experiment_dir, logger):
    model_path = os.path.join(experiment_dir, 'svm_model.pkl')
    logger.info("Load model from " + str(model_path))
    f = open(model_path, 'rb')
    classifier = pickle.load(f)
    f.close()

    pred = np.array(classifier.predict(feature_vectors))
    pred = [str(i) for i in pred]

    test_labels = np.array(labels)
    test_labels = [str(i) for i in test_labels]

    logger.info("Accuracy:" + str(accuracy(test_labels, pred)))

    logger.info("Unweighted average recall:" + str(recall_score(test_labels, pred, average='macro')))

    cm = ConfusionMatrix(test_labels, pred)

    logger.info("Confusion Matrix:\n" + str(cm))

