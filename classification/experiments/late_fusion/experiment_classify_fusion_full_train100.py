from classification.audio import decision_tree, svm, max_prob_classifier
from models import DNN
from classification.text import naive_bayes
import classification.util.data_loader_txt as data_loader
from classification.util.experiments_util import *
from classification.util.global_vars import *

experiments_folder = os.path.join(ROOT_FOLDER, "IEMOCAP_fusion//experiments")

audio_train = os.path.join(ROOT_FOLDER, "features//late_fusion//audio_train.txt")
audio_dev = os.path.join(ROOT_FOLDER, "features//late_fusion//audio_dev.txt")
audio_test = os.path.join(ROOT_FOLDER, "features//late_fusion//audio_test.txt")

text_train = os.path.join(ROOT_FOLDER, "features//late_fusion//text_train.txt")
text_dev = os.path.join(ROOT_FOLDER, "features//late_fusion//text_dev.txt")
text_test = os.path.join(ROOT_FOLDER, "features//late_fusion//text_test.txt")

label_to_id = {
    "hap":"0",
    "exc":"0",
    "sad":"1",
    "ang":"2",
    "neu":"3",
}



experiment_dir, logger = create_experiment(experiments_folder, label_to_id, "classify_fusion_full_no_train_split", use_timestamp=True)

DNN.train(audio_train, experiment_dir, label_to_id, logger)
DNN.test(audio_dev, experiment_dir, label_to_id, logger)

naive_bayes.train(text_train, experiment_dir, label_to_id, logger)
naive_bayes.test(text_dev, experiment_dir, label_to_id, logger)

#extract late_fusion probabilities for train:
probabilities_text = naive_bayes.eval_get_probability_scores(text_train, experiment_dir, label_to_id, logger)
probabilities_audio = DNN.eval_get_probabilities(audio_train, experiment_dir, label_to_id, logger)
ids, labels = get_ids_and_labels(text_train, label_to_id)

joined = join_ids_labels_probs(ids, labels, probabilities_text, probabilities_audio)
fusion_path_train = os.path.join(experiment_dir, "fusion_train_30.txt")

with open(fusion_path_train, "w") as f:
    f.write(joined)

#extract late_fusion probabilities for dev:
probabilities_text = naive_bayes.eval_get_probability_scores(text_dev, experiment_dir, label_to_id, logger)
probabilities_audio = DNN.eval_get_probabilities(audio_dev, experiment_dir, label_to_id, logger)
ids, labels = get_ids_and_labels(text_dev, label_to_id)

joined = join_ids_labels_probs(ids, labels, probabilities_text, probabilities_audio)
fusion_path_dev = os.path.join(experiment_dir, "fusion_dev.txt")

with open(fusion_path_dev, "w") as f:
    f.write(joined)

#extract late_fusion probabilities for test:
probabilities_text = naive_bayes.eval_get_probability_scores(text_test, experiment_dir, label_to_id, logger)
probabilities_audio = DNN.eval_get_probabilities(audio_test, experiment_dir, label_to_id, logger)
ids, labels = get_ids_and_labels(text_test, label_to_id)

joined = join_ids_labels_probs(ids, labels, probabilities_text, probabilities_audio)
fusion_path_test = os.path.join(experiment_dir, "fusion_test.txt")

with open(fusion_path_test, "w") as f:
    f.write(joined)

train_vectors, train_labels = data_loader.get_train_data(fusion_path_train, label_to_id, experiment_dir, False, logger)

dev_vectors, dev_labels = data_loader.get_test_data(fusion_path_dev, label_to_id, experiment_dir, False, logger)

decision_tree.train(train_labels, train_vectors, experiment_dir, logger)
decision_tree.test(dev_labels, dev_vectors, experiment_dir, logger)

svm.train(train_labels, train_vectors, experiment_dir, logger)
svm.test(dev_labels, dev_vectors, experiment_dir, logger)

max_prob_classifier.test(dev_labels, dev_vectors, logger)