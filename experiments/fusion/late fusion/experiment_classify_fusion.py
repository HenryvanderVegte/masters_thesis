from experiments.text.naive_bayes import naive_bayes
from experiments.audio import svm
from experiments.util.experiments_util import *

FUSION_TRAIN = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_fusion//features//fusion//train_30_fusion.txt"
FUSION_DEV = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_fusion//features//fusion//dev_fusion.txt"
AUDIO_TEST = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_fusion//features//fusion//test_fusion.txt"

EXPERIMENTS_FOLDER = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_fusion//experiments//"

label_to_id = {
    "hap":"0",
    "exc":"0",
    "sad":"1",
    "ang":"2",
    "neu":"3",
}

experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, label_to_id, "classify fusion experiment")

svm.train(FUSION_TRAIN, experiment_dir, label_to_id, logger)
svm.test(FUSION_DEV, experiment_dir, label_to_id, logger)