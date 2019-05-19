import numpy as np
from sklearn.svm import SVC
import pickle, os
from classification.util.experiments_util import get_metrics

def train(labels, feature_vectors, experiment_dir, logger):
    logger.info("############ Training SVM classifier. ########## \n\n" )
    classifier = SVC(gamma='scale', decision_function_shape='ovo')
    classifier.fit(feature_vectors, labels)

    model_path = os.path.join(experiment_dir, 'svm_model.pkl')
    f = open(model_path, 'wb')
    pickle.dump(classifier, f)
    f.close()

    logger.info("Completed training. Saved model to " + model_path)

def test(labels, feature_vectors, experiment_dir, logger):
    logger.info("############ Testing SVM classifier. ########## \n\n" )
    model_path = os.path.join(experiment_dir, 'svm_model.pkl')
    logger.info("Load model from " + str(model_path))
    f = open(model_path, 'rb')
    classifier = pickle.load(f)
    f.close()

    pred = np.array(classifier.predict(feature_vectors))
    pred = [str(i) for i in pred]

    labels = np.array(labels)
    labels = [str(i) for i in labels]

    logger.info(get_metrics(labels, pred))
