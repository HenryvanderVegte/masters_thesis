from experiments.audio import dnn, svm
from experiments.text import naive_bayes
from experiments.util.experiments_util import *

AUDIO_TRAIN_70 = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_fusion//features//audio//train_70.txt"
AUDIO_TRAIN_30 = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_fusion//features//audio//train_30.txt"
AUDIO_DEV = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_fusion//features//audio//dev.txt"
AUDIO_TEST = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_fusion//features//audio//test.txt"

TEXT_TRAIN_70 = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_fusion//features//text//train_70.txt"
TEXT_TRAIN_30 = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_fusion//features//text//train_30.txt"
TEXT_DEV = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_fusion//features//text//dev.txt"
TEXT_TEST = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_fusion//features//text//test.txt"

EXPERIMENTS_FOLDER = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_fusion//experiments//"

label_to_id = {
    "hap":"0",
    "exc":"0",
    "sad":"1",
    "ang":"1",
    "neu":"2",
}

experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, label_to_id, "classify_fusion_full_3_labels", use_timestamp=True)

dnn.train(AUDIO_TRAIN_70, experiment_dir, label_to_id, logger)
dnn.test(AUDIO_DEV, experiment_dir, label_to_id, logger)

naive_bayes.train(TEXT_TRAIN_70, experiment_dir, label_to_id, logger)
naive_bayes.test(TEXT_DEV, experiment_dir, label_to_id, logger)

#extract fusion probabilities for train:
probabilities_text = naive_bayes.eval_get_probability_scores(TEXT_TRAIN_30, experiment_dir, label_to_id, logger)
probabilities_audio = dnn.eval_get_probabilities(AUDIO_TRAIN_30, experiment_dir, label_to_id, logger)
ids, labels = get_ids_and_labels(TEXT_TRAIN_30, label_to_id)

joined = join_ids_labels_probs(ids, labels, probabilities_text, probabilities_audio)
fusion_path_train = os.path.join(experiment_dir, "train_30_fusion.txt")

with open(fusion_path_train, "w") as f:
    f.write(joined)

#extract fusion probabilities for dev:
probabilities_text = naive_bayes.eval_get_probability_scores(TEXT_DEV, experiment_dir, label_to_id, logger)
probabilities_audio = dnn.eval_get_probabilities(AUDIO_DEV, experiment_dir, label_to_id, logger)
ids, labels = get_ids_and_labels(TEXT_DEV, label_to_id)

joined = join_ids_labels_probs(ids, labels, probabilities_text, probabilities_audio)
fusion_path_dev = os.path.join(experiment_dir, "dev_fusion.txt")

with open(fusion_path_dev, "w") as f:
    f.write(joined)

#extract fusion probabilities for test:
probabilities_text = naive_bayes.eval_get_probability_scores(TEXT_TEST, experiment_dir, label_to_id, logger)
probabilities_audio = dnn.eval_get_probabilities(AUDIO_TEST, experiment_dir, label_to_id, logger)
ids, labels = get_ids_and_labels(TEXT_TEST, label_to_id)

joined = join_ids_labels_probs(ids, labels, probabilities_text, probabilities_audio)
fusion_path_test = os.path.join(experiment_dir, "test_fusion.txt")

with open(fusion_path_test, "w") as f:
    f.write(joined)

svm.train_from_file(fusion_path_train, experiment_dir, label_to_id, logger)
svm.test_from_file(fusion_path_dev, experiment_dir, label_to_id, logger)

