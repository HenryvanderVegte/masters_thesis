from classification.audio import svm
from classification.util.experiments_util import *
from classification.util.global_vars import *
import os

experiments_folder = os.path.join(ROOT_FOLDER, "IEMOCAP_fusion//experiments")

fusion_train = os.path.join(ROOT_FOLDER, "features//fusion//fusion_train_30.txt")
fusion_dev = os.path.join(ROOT_FOLDER, "features//fusion//fusion_dev.txt")

label_to_id = {
    "hap":"0",
    "exc":"0",
    "sad":"1",
    "ang":"2",
    "neu":"3",
}

experiment_dir, logger = create_experiment(experiments_folder, label_to_id, "classify_fusion_svm", use_timestamp=True)

svm.train_from_file(fusion_train, experiment_dir, label_to_id, logger)
svm.test_from_file(fusion_dev, experiment_dir, label_to_id, logger)