import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle, os
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus
from utils.experiments_util import get_metrics_str

def train(train_dataset, dev_dataset, experiment_path, logger):
    train_features = np.array(train_dataset.tensors[0])
    train_labels = np.ravel(np.array(train_dataset.tensors[1]))
    dev_features = np.array(dev_dataset.tensors[0])
    dev_labels = np.ravel(np.array(dev_dataset.tensors[1]))

    classifier = DecisionTreeClassifier(max_leaf_nodes=16, random_state=0)
    classifier.fit(train_features, train_labels)

    model_path = os.path.join(experiment_path, 'decision_tree.pkl')
    f = open(model_path, 'wb')
    pickle.dump(classifier, f)
    f.close()

    #Print out decision tree as image
    dot_data = StringIO()
    export_graphviz(classifier, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    img = graph.create_png()
    img_path = os.path.join(experiment_path, 'decision_tree.png')
    with open(img_path, "wb") as png:
        png.write(img)

    pred = np.array(classifier.predict(dev_features))
    pred = [str(i) for i in pred]
    dev_labels = [str(int(i)) for i in dev_labels]

    logger.info(get_metrics_str(dev_labels, pred))