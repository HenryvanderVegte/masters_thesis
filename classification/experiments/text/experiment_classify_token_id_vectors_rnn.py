import classification.util.data_loader_pickle as data_loader
from classification.util.experiments_util import *
from classification.util.global_vars import *
from classification.text import rnn_embeddings
from classification.util.text_preprocessing import create_sequence_dataset_with_pad_val
import numpy as np

train_labels = os.path.join(ROOT_FOLDER, "IEMOCAP//features//text//embeddings//token_to_id//train_token_id.txt")
train_token_id_vectors = os.path.join(ROOT_FOLDER, "IEMOCAP//features//text//embeddings//token_to_id//train_token_id.npy")

dev_labels = os.path.join(ROOT_FOLDER, "IEMOCAP//features//text//embeddings//token_to_id//dev_token_id.txt")
dev_token_id_vectors = os.path.join(ROOT_FOLDER, "IEMOCAP//features//text//embeddings//token_to_id//dev_token_id.npy")

token_id_dict = os.path.join(ROOT_FOLDER, "IEMOCAP//features//text//embeddings//token_to_id//vector_dict.npy")

experiments_folder = os.path.join(ROOT_FOLDER, "IEMOCAP_txt//experiments")

label_to_id = {
    "hap":"0",
    "exc":"0",
    "sad":"1",
    "ang":"2",
    "neu":"3",
}

params = {
    "sequence_length":25,
    "embedding_size":100,
    "batch_size": 16,
    "hidden_size": 15,
    "drop_prob": 0.6,
    "layers": 2,
    "epochs": 1000,
}

experiment_dir, logger = create_experiment(experiments_folder, label_to_id, "classify_token_id_vectors_rnn", use_timestamp=True)

token_to_id_dict = np.load(token_id_dict).item()

#this is always the last id in our dict and therefore the size
params["vocab_size"] = len(token_to_id_dict)

train_labels, train_features = data_loader.load_dict_from_binary(train_labels, train_token_id_vectors, label_to_id)
train_dataset = create_sequence_dataset_with_pad_val(train_features, train_labels, params["sequence_length"], token_to_id_dict["EOU"])

dev_labels, dev_features = data_loader.load_dict_from_binary(dev_labels, dev_token_id_vectors, label_to_id)
dev_dataset = create_sequence_dataset_with_pad_val(dev_features, dev_labels, params["sequence_length"], token_to_id_dict["EOU"])

rnn_embeddings.train(train_dataset, dev_dataset, experiment_dir, label_to_id, logger, params)
