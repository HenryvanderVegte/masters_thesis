import classification.util.data_loader_pickle as data_loader
from classification.util.experiments_util import *
from classification.util.global_vars import *
from classification.text import rnn_embeddings
from classification.text.word_embeddings import *
from classification.util.text_preprocessing import create_sequence_dataset_with_pad_val
import numpy as np

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

params = {
    "sequence_length":40,
    "embedding_size":20,
    "batch_size": 16,
    "hidden_size": 10,
    "drop_prob": 0.6,
    "layers": 2,
    "epochs": 1000,
    "vocab_size": 3000
}

experiment_dir, logger = create_experiment(experiments_folder, label_to_id, "extract_token_id_vectors_with_time", use_timestamp=True)

# extract features:
token_id_dict = build_token_to_id_dict(train_txt, 5)
np.save(os.path.join(experiment_dir, 'vector_dict.npy'), dict)

train_token_id_path = os.path.join(experiment_dir, 'train_token_id.npy')
train_labels_path = os.path.join(experiment_dir, 'train_token_id.txt')
extract_token_to_id_vectors_with_time(train_txt, token_id_dict, train_labels_path, train_token_id_path)

dev_token_id_path = os.path.join(experiment_dir, 'dev_token_id.npy')
dev_labels_path = os.path.join(experiment_dir, 'dev_token_id.txt')
extract_token_to_id_vectors_with_time(dev_txt, token_id_dict, dev_labels_path, dev_token_id_path)

train_labels, train_features = data_loader.load_dict_from_binary(train_labels_path, train_token_id_path, label_to_id)
train_dataset = create_sequence_dataset_with_pad_val(train_features, train_labels, params["sequence_length"], token_id_dict["EOU"])

dev_labels, dev_features = data_loader.load_dict_from_binary(dev_labels_path, dev_token_id_path, label_to_id)
dev_dataset = create_sequence_dataset_with_pad_val(dev_features, dev_labels, params["sequence_length"], token_id_dict["EOU"])

rnn_embeddings.train(train_dataset, dev_dataset, experiment_dir, label_to_id, logger, params)
