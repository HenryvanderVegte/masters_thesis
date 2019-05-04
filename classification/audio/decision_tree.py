import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle, os
from sklearn.metrics import recall_score
from nltk.metrics import ConfusionMatrix, accuracy
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus



def train(labels, feature_vectors, experiment_dir, logger):
    logger.info("Training Decision Tree classifier.")
    classifier = DecisionTreeClassifier(max_leaf_nodes=16, random_state=0)
    classifier.fit(feature_vectors, labels)

    model_path = os.path.join(experiment_dir, 'dtc_model.pkl')
    f = open(model_path, 'wb')
    pickle.dump(classifier, f)
    f.close()

    logger.info("Completed training. Saved model to " + model_path)

    dot_data = StringIO()

    export_graphviz(classifier, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    img = graph.create_png()
    img_path = os.path.join(experiment_dir, 'dct.png')
    with open(img_path, "wb") as png:
        png.write(img)

def test(labels, feature_vectors, experiment_dir, logger):
    model_path = os.path.join(experiment_dir, 'dtc_model.pkl')
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