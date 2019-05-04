from classification.util.experiments_util import *
from classification.audio import dnn

EXPERIMENTS_FOLDER = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//classification//"

TRAIN_TXT = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//split//train.txt"
DEV_TXT = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//split//dev.txt"
TEST_TXT = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//split//test.txt"

label_to_id = {
    "hap":"0",
    "sad":"1",
    "ang":"2",
    "neu":"3",
}

experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, label_to_id, "classify_dnn_4_labels", use_timestamp=True)

dnn.train(TRAIN_TXT, experiment_dir, label_to_id, logger)

dnn.test(DEV_TXT, experiment_dir, label_to_id, logger)