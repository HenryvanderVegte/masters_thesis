import classification.util.data_loader_pickle as data_loader
from classification.util.experiments_util import *
from classification.util.global_vars import *
from classification.text import rnn_pretrained_embeddings
from classification.util.text_preprocessing import create_sequence_dataset

TRAIN_EMBEDDINGS_LABELS = os.path.join(ROOT_FOLDER, "IEMOCAP_txt//embeddings//2-dim//train_embeddings.txt")
TRAIN_EMBEDDINGS = os.path.join(ROOT_FOLDER, "IEMOCAP_txt//embeddings//2-dim//train_embeddings.npy")

DEV_EMBEDDINGS_LABELS = os.path.join(ROOT_FOLDER, "IEMOCAP_txt//embeddings//2-dim//dev_embeddings.txt")
DEV_EMBEDDINGS = os.path.join(ROOT_FOLDER, "IEMOCAP_txt//embeddings//2-dim//dev_embeddings.npy")

EXPERIMENTS_FOLDER = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//classification//"

label_to_id = {
    "hap":"0",
    "exc":"0",
    "sad":"1",
    "ang":"2",
    "neu":"3",
}

params = {
    "sequence_length":30,
    "layers": 2,
    "epochs": 200,
}

experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, label_to_id, "classify_word_embeddings_dnn", use_timestamp=True)

train_labels, train_features = data_loader.load_dict_from_binary(TRAIN_EMBEDDINGS_LABELS, TRAIN_EMBEDDINGS, label_to_id)
train_dataset = create_sequence_dataset(train_features, train_labels, params["sequence_length"])

dev_labels, dev_features = data_loader.load_from_binary(DEV_EMBEDDINGS_LABELS, DEV_EMBEDDINGS, label_to_id)
dev_dataset = create_sequence_dataset(dev_features, dev_labels, params["sequence_length"])

rnn_pretrained_embeddings.train(train_dataset, dev_dataset, experiment_dir, label_to_id, logger, params)


