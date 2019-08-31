from models import DNN
from classification.text import naive_bayes
from classification.util.experiments_util import *
from classification.util.global_vars import *

experiments_folder = os.path.join(ROOT_FOLDER, "IEMOCAP_fusion//experiments")

audio_train_70 = os.path.join(ROOT_FOLDER, "features//late_fusion//audio_train_70.txt")
audio_train_30 = os.path.join(ROOT_FOLDER, "features//late_fusion//audio_train_30.txt")
audio_dev = os.path.join(ROOT_FOLDER, "features//late_fusion//audio_dev.txt")
audio_test = os.path.join(ROOT_FOLDER, "features//late_fusion//audio_test.txt")

text_train_70 = os.path.join(ROOT_FOLDER, "features//late_fusion//text_train_70.txt")
text_train_30 = os.path.join(ROOT_FOLDER, "features//late_fusion//text_train_30.txt")
text_dev = os.path.join(ROOT_FOLDER, "features//late_fusion//text_dev.txt")
text_test = os.path.join(ROOT_FOLDER, "features//late_fusion//text_test.txt")

label_to_id = {
    "hap":"0",
    "exc":"0",
    "sad":"1",
    "ang":"2",
    "neu":"3",
}

experiment_dir, logger = create_experiment(experiments_folder, label_to_id, "extract_late_fusion_features", use_timestamp=True)

DNN.train(audio_train_70, experiment_dir, label_to_id, logger)
DNN.test(audio_dev, experiment_dir, label_to_id, logger)

naive_bayes.train(text_train_70, experiment_dir, label_to_id, logger)
naive_bayes.test(text_dev, experiment_dir, label_to_id, logger)

#extract late_fusion probabilities for train:
probabilities_text = naive_bayes.eval_get_probability_scores(text_train_30, experiment_dir, label_to_id, logger)
probabilities_audio = DNN.eval_get_probabilities(audio_train_30, experiment_dir, label_to_id, logger)
ids, labels = get_ids_and_labels(text_train_30, label_to_id)

joined = join_ids_labels_probs(ids, labels, probabilities_text, probabilities_audio)
fusion_path = os.path.join(experiment_dir, "fusion_train_30.txt")

with open(fusion_path, "w") as f:
    f.write(joined)

#extract late_fusion probabilities for dev:
probabilities_text = naive_bayes.eval_get_probability_scores(text_dev, experiment_dir, label_to_id, logger)
probabilities_audio = DNN.eval_get_probabilities(audio_dev, experiment_dir, label_to_id, logger)
ids, labels = get_ids_and_labels(text_dev, label_to_id)

joined = join_ids_labels_probs(ids, labels, probabilities_text, probabilities_audio)
fusion_path = os.path.join(experiment_dir, "fusion_dev.txt")

with open(fusion_path, "w") as f:
    f.write(joined)

#extract late_fusion probabilities for test:
probabilities_text = naive_bayes.eval_get_probability_scores(text_test, experiment_dir, label_to_id, logger)
probabilities_audio = DNN.eval_get_probabilities(audio_test, experiment_dir, label_to_id, logger)
ids, labels = get_ids_and_labels(text_test, label_to_id)

joined = join_ids_labels_probs(ids, labels, probabilities_text, probabilities_audio)
fusion_path = os.path.join(experiment_dir, "fusion_test.txt")

with open(fusion_path, "w") as f:
    f.write(joined)
