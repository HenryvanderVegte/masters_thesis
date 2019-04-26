from experiments.text.naive_bayes import naive_bayes
from experiments.audio.dnn_grouped import dnn
import os
from experiments.util.experiments_util import create_experiment, get_ids_and_labels

AUDIO_TRAIN_70 = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_fusion//features//audio//train_70.txt"
AUDIO_TRAIN_30 = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_fusion//features//audio//train_30.txt"
AUDIO_DEV = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_fusion//features//audio//dev.txt"

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

experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, label_to_id, "late fusion experiment")

naive_bayes.train(TEXT_TRAIN_70, experiment_dir, label_to_id, logger)
naive_bayes.test(TEXT_DEV, experiment_dir, label_to_id, logger)

probabilities_text = naive_bayes.eval_get_probability_scores(TEXT_TRAIN_30, experiment_dir, label_to_id, logger)

dnn.train(AUDIO_TRAIN_70, experiment_dir, label_to_id, logger)
dnn.test(AUDIO_DEV, experiment_dir)

probabilities_audio = dnn.eval_get_probabilities(AUDIO_TRAIN_30, experiment_dir, label_to_id, logger)

ids_labels = get_ids_and_labels(TEXT_TRAIN_30)

fusion_train_path = os.path.join(experiment_dir, "train_30_fusion.txt")
out = ""
for i in range(0, len(ids_labels)):
    out += ids_labels[i] + "\t"
    text_probs = probabilities_text[i]
    for text_prob in text_probs:
        out += text_prob + "\t"

    audio_probs = probabilities_audio[i]
    for audio_prob in audio_probs:
        out += audio_prob + "\t"
    out += "\n"

