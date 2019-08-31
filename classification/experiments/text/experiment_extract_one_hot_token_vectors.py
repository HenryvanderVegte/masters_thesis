from utils.experiments_util import *
from classification.util.global_vars import *
from classification.text.word_embeddings import *
import os


experiments_folder = os.path.join(ROOT_FOLDER, "IEMOCAP_txt//experiments")


train_txt = os.path.join(ROOT_FOLDER, "IEMOCAP//features//text//utterances//train.txt")
dev_txt = os.path.join(ROOT_FOLDER, "IEMOCAP//features//text//utterances//dev.txt")
test_txt = os.path.join(ROOT_FOLDER, "IEMOCAP//features//text//utterances//test.txt")

label_to_id = {
    "hap":"0",
    "exc":"0",
    "sad":"1",
    "ang":"2",
    "neu":"3",
}

experiment_dir, logger = create_experiment(experiments_folder, label_to_id, "extract_one_hot_vectors_with_time", use_timestamp=True)

dict = build_one_hot_dict(train_txt, 3000)

np.save(os.path.join(experiment_dir, 'vector_dict.npy'), dict)

train_one_hot_path = os.path.join(experiment_dir, 'train_one_hot.npy')
train_labels_path = os.path.join(experiment_dir, 'train_one_hot.txt')
extract_one_hot_vectors_with_time(train_txt, dict, train_labels_path, train_one_hot_path)

dev_one_hot_path = os.path.join(experiment_dir, 'dev_one_hot.npy')
dev_labels_path = os.path.join(experiment_dir, 'dev_one_hot.txt')
extract_one_hot_vectors_with_time(dev_txt, dict, dev_labels_path, dev_one_hot_path)

test_one_hot_path = os.path.join(experiment_dir, 'test_one_hot.npy')
test_labels_path = os.path.join(experiment_dir, 'test_one_hot.txt')
extract_one_hot_vectors_with_time(test_txt, dict, test_labels_path, test_one_hot_path)