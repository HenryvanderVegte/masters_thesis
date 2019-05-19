import classification.util.data_loader_pickle as data_loader
from classification.util.experiments_util import *
from classification.util.global_vars import *
from classification.util.rnn_utils import *
from classification.text import rnn_pretrained_embeddings
from classification.util.text_preprocessing import create_sequence_dataset

TRAIN_EMBEDDINGS_LABELS = os.path.join(ROOT_FOLDER, "IEMOCAP_txt//embeddings//2-dim//train_embeddings.txt")
TRAIN_EMBEDDINGS = os.path.join(ROOT_FOLDER, "IEMOCAP_txt//embeddings//2-dim//train_embeddings.npy")

DEV_EMBEDDINGS_LABELS = os.path.join(ROOT_FOLDER, "IEMOCAP_txt//embeddings//2-dim//dev_embeddings.txt")
DEV_EMBEDDINGS = os.path.join(ROOT_FOLDER, "IEMOCAP_txt//embeddings//2-dim//dev_embeddings.npy")

EXPERIMENTS_FOLDER = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//experiments//"

class_groups = {
    "hap":"0",
    "exc":"0",
    "sad":"1",
    "ang":"2",
    "neu":"3",
}

params = {
    "max_sequence_length": 50,
    "batch_size": 16,
    "hidden_size": 16,
    "drop_prob": 0.5,
    "layers": 2,
    "epochs": 1000,
    "log_x_epochs": 8,
}

experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, class_groups, "classify_word_embeddings_rnn", use_timestamp=True)
params["labels_size"] = len(set(list(class_groups.values())))

dev_labels, dev_features = data_loader.load_dict_from_binary(DEV_EMBEDDINGS_LABELS, DEV_EMBEDDINGS, class_groups)
dev_dataset, id_to_name = create_sequence_dataset(dev_features, dev_labels, params["max_sequence_length"])

params["embedding_dim"] = dev_dataset.tensors[0][0].size()[1]

train_labels, train_features = data_loader.load_dict_from_binary(TRAIN_EMBEDDINGS_LABELS, TRAIN_EMBEDDINGS, class_groups)
train_dataset, _ = create_sequence_dataset(train_features, train_labels, params["max_sequence_length"])

model = rnn_pretrained_embeddings.PretrainedEmbeddingsLSTM(params)

train(train_dataset, dev_dataset, id_to_name, experiment_dir, model, logger, params)
