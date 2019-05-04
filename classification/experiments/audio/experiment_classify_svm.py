from classification.util.experiments_util import *
from classification.audio import svm

EXPERIMENTS_FOLDER = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//classification//"

TRAIN_TXT = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//split//train.txt"
DEV_TXT = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//split//dev.txt"
TEST_TXT = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//split//test.txt"

label_to_id = {
    "hap":"0",
    "exc":"0",
    "sad":"1",
    "ang":"1",
    "neu":"2",
}

experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, label_to_id, "classify_svm_3_labels", use_timestamp=True)

svm.train_from_file(TRAIN_TXT, experiment_dir, label_to_id, logger)

svm.test_from_file(DEV_TXT, experiment_dir, label_to_id, logger)