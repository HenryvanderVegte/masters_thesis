from experiments.util.experiments_util import *
from experiments.audio import dnn

EXPERIMENTS_FOLDER = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//experiments//"

TRAIN_TXT = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//split//train.txt"
DEV_TXT = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//split//dev.txt"
TEST_TXT = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_audio//split//test.txt"

label_to_id = {
    "hap":"0",
    "exc":"0",
    "sad":"1",
    "fru":"1",
    "ang":"1",
    "fea":"1",
    "dis":"1",
    "neu":"2",
}

experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, label_to_id, "classify_dnn_3_labels", use_timestamp=True)

dnn.train(TRAIN_TXT, experiment_dir, label_to_id, logger)

dnn.test(DEV_TXT, experiment_dir, label_to_id, logger)