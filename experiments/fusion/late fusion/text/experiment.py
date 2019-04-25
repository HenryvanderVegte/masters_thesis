from experiments.text.naive_bayes import naive_bayes
import experiments.audio.dnn_grouped
import os
from experiments.util.experiments_util import create_experiment

AUDIO_TRAIN_70 = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_fusion//features//audio//train_70.txt"
AUDIO_TRAIN_30 = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_fusion//features//audio//train_30.txt"

TEXT_TRAIN_70 = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_fusion//features//text//train_70.txt"
TEXT_TRAIN_30 = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_fusion//features//text//train_30.txt"
TEXT_DEV = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_fusion//features//text//dev.txt"

EXPERIMENTS_FOLDER = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_fusion//experiments//"

label_to_id = {
    "hap":"0",
    "exc":"0",
    "sad":"1",
    "ang":"2",
    "neu":"3",
}

experiment_dir, logfile = create_experiment(EXPERIMENTS_FOLDER, label_to_id, "late fusion experiment")

naive_bayes.train(TEXT_TRAIN_70, experiment_dir, label_to_id, logfile)
naive_bayes.test(TEXT_DEV, experiment_dir, label_to_id, logfile)

probabilities = naive_bayes.eval_get_probability_scores(TEXT_TRAIN_30, experiment_dir, label_to_id, logfile)

print(probabilities)