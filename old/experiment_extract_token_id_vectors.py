from utils.experiments_util import *
from classification.util.global_vars import *
from old.word_embeddings import *
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

experiment_dir, logger = create_experiment(experiments_folder, label_to_id, "extract_token_id_vectors_with_time", use_timestamp=True)

dict = build_token_to_id_dict(train_txt, 5)

np.save(os.path.join(experiment_dir, 'vector_dict.npy'), dict)

train_token_id_path = os.path.join(experiment_dir, 'train_token_id.npy')
train_labels_path = os.path.join(experiment_dir, 'train_token_id.txt')
extract_token_to_id_vectors_with_time(train_txt, dict, train_labels_path, train_token_id_path)

dev_token_id_path = os.path.join(experiment_dir, 'dev_token_id.npy')
dev_labels_path = os.path.join(experiment_dir, 'dev_token_id.txt')
extract_token_to_id_vectors_with_time(dev_txt, dict, dev_labels_path, dev_token_id_path)

test_token_id_path = os.path.join(experiment_dir, 'test_token_id.npy')
test_labels_path = os.path.join(experiment_dir, 'test_token_id.txt')
extract_token_to_id_vectors_with_time(test_txt, dict, test_labels_path, test_token_id_path)