from classification.audio import decision_tree
from classification.util.experiments_util import *
import classification.util.data_loader_txt as data_loader
from classification.util.global_vars import *
import os

experiments_folder = os.path.join(ROOT_FOLDER, "IEMOCAP_fusion//experiments")

fusion_train = os.path.join(ROOT_FOLDER, "features//late_fusion//fusion_train_30.txt")
fusion_dev = os.path.join(ROOT_FOLDER, "features//late_fusion//fusion_dev.txt")

label_to_id = {
    "hap":"0",
    "exc":"0",
    "sad":"1",
    "ang":"2",
    "neu":"3",
}


experiment_dir, logger = create_experiment(experiments_folder, label_to_id, "classify_fusion_dtc", use_timestamp=True)

train_vectors, train_labels = data_loader.get_train_data(fusion_train, label_to_id, experiment_dir, False, logger)

dev_vectors, dev_labels = data_loader.get_test_data(fusion_dev, label_to_id, experiment_dir, False, logger)

decision_tree.train(train_labels, train_vectors, experiment_dir, logger)
decision_tree.test(dev_labels, dev_vectors, experiment_dir, logger)