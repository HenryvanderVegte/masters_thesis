from experiments.util.experiments_util import *
from experiments.text import naive_bayes

EXPERIMENTS_FOLDER = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//experiments//"

TRAIN_TXT = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//split//train.txt"
DEV_TXT = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//split//dev.txt"
TEST_TXT = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//split//test.txt"

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

experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, label_to_id, "classify_naive_bayes", use_timestamp=True)


naive_bayes.train(TRAIN_TXT, experiment_dir, label_to_id, logger)

naive_bayes.test(DEV_TXT, experiment_dir, label_to_id, logger)