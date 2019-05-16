import classification.util.data_loader_pickle as data_loader
from classification.util.experiments_util import *
from classification.util.global_vars import *
from classification.text import rnn_embeddings
from classification.util.text_preprocessing import create_sequence_dataset

train_labels = os.path.join(ROOT_FOLDER, "IEMOCAP//features//text//embeddings//one-hot//train_one_hot.txt")
train_one_hot_vectors = os.path.join(ROOT_FOLDER, "IEMOCAP//features//text//embeddings//one-hot//train_one_hot.npy")

dev_labels = os.path.join(ROOT_FOLDER, "IEMOCAP//features//text//embeddings//one-hot//dev_one_hot.txt")
dev_one_hot_vectors = os.path.join(ROOT_FOLDER, "IEMOCAP//features//text//embeddings//one-hot//dev_one_hot.npy")

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

experiment_dir, logger = create_experiment(experiments_folder, label_to_id, "classify_one_hot_vectors_rnn", use_timestamp=True)

train_labels, train_features = data_loader.load_dict_from_binary(train_labels, train_one_hot_vectors, label_to_id)
train_dataset = create_sequence_dataset(train_features, train_labels, params["sequence_length"])

dev_labels, dev_features = data_loader.load_dict_from_binary(dev_labels, dev_one_hot_vectors, label_to_id)
dev_dataset = create_sequence_dataset(dev_features, dev_labels, params["sequence_length"])

rnn_embeddings.train(train_dataset, dev_dataset, experiment_dir, label_to_id, logger, params)
