from classification.audio import max_prob_classifier
from utils.experiments_util import *
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


experiment_dir, logger = create_experiment(experiments_folder, label_to_id, "classify_fusion_max_prob", use_timestamp=True)

dev_vectors, dev_labels = data_loader.get_test_data(fusion_dev, label_to_id, experiment_dir, False, logger)

max_prob_classifier.test(dev_labels, dev_vectors, logger)