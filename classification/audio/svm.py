import numpy as np
from sklearn.svm import SVC
import pickle, os
from utils.experiments_util import get_metrics_str

def train(train_dataset, dev_dataset, experiment_dir, logger):
    features = np.array(train_dataset.tensors[0])
    labels = np.ravel(np.array(train_dataset.tensors[1]))
    classifier = SVC(gamma='scale', decision_function_shape='ovo')
    classifier.fit(features, labels)

    model_path = os.path.join(experiment_dir, 'svm_model.pkl')
    f = open(model_path, 'wb')
    pickle.dump(classifier, f)
    f.close()

    dev_features = np.array(dev_dataset.tensors[0])
    dev_labels = np.ravel(np.array(dev_dataset.tensors[1]))

    pred = np.array(classifier.predict(dev_features))
    pred = [str(int(i)) for i in pred]

    dev_labels = [str(int(i)) for i in dev_labels]

    logger.info(get_metrics_str(dev_labels, pred))

    logger.info("Completed training. Saved model to " + model_path)
