import classification.util.data_loader_pickle as data_loader
from classification.util.experiments_util import *
from utils.rnn_utils import *
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

class_groups = {
    "hap":"0",
    "exc":"0",
    "sad":"1",
    "ang":"2",
    "neu":"3",
}

params = {
    "max_sequence_length":50,
    "embedding_size":16,
    "batch_size": 16,
    "hidden_size": 8,
    "drop_prob": 0.4,
    "layers": 2,
    "epochs": 1000,
    "log_x_epochs": 6,
}

experiment_dir, logger = create_experiment(experiments_folder, class_groups, "classify_token_id_vectors_rnn", use_timestamp=True)
token_to_id_dict = np.load(token_id_dict).item()

params["vocab_size"] = len(token_to_id_dict)
params["labels_size"] = len(set(list(class_groups.values())))

train_labels, train_features = data_loader.load_dict_from_binary(train_labels, train_token_id_vectors, class_groups)
train_dataset, _ = create_sequence_dataset_with_pad_val(train_features, train_labels, params["max_sequence_length"], token_to_id_dict["EOU"])

dev_labels, dev_features = data_loader.load_dict_from_binary(dev_labels, dev_token_id_vectors, class_groups)
dev_dataset, id_to_name = create_sequence_dataset_with_pad_val(dev_features, dev_labels, params["max_sequence_length"], token_to_id_dict["EOU"])

model = rnn_embeddings.SelfLearnedEmbeddingsLSTM(params)

train(train_dataset, dev_dataset, id_to_name, experiment_dir, model, logger, params)