from classification.util.experiments_util import *
from classification.audio import svm
from classification.util.global_vars import *

experiments_folder = os.path.join(ROOT_FOLDER, "IEMOCAP_audio//experiments")

train_txt = os.path.join(ROOT_FOLDER, "features//audio//train.txt")
dev_txt = os.path.join(ROOT_FOLDER, "features//audio//dev.txt")

label_to_id = {
    "hap":"0",
    "exc":"0",
    "sad":"1",
    "ang":"1",
    "neu":"2",
}

experiment_dir, logger = create_experiment(experiments_folder, label_to_id, "classify_svm_3_labels", use_timestamp=True)

svm.train_from_file(train_txt, experiment_dir, label_to_id, logger)

svm.test_from_file(dev_txt, experiment_dir, label_to_id, logger)