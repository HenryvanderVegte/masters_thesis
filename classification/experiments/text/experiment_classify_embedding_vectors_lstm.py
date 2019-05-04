from classification.util.data_loader import *
from classification.util.experiments_util import *
from classification.util.global_vars import *
from classification.text import dnn

TRAIN_EMBEDDINGS_LABELS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//embeddings//2-dim//train_embeddings.txt"
TRAIN_EMBEDDINGS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//embeddings//2-dim//train_embeddings.npy"

DEV_EMBEDDINGS_LABELS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//embeddings//2-dim//dev_embeddings.txt"
DEV_EMBEDDINGS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//embeddings//2-dim//dev_embeddings.npy"

TEST_EMBEDDINGS_LABELS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//embeddings//2-dim//test_embeddings.txt"
TEST_EMBEDDINGS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//embeddings//2-dim//test_embeddings.npy"

EXPERIMENTS_FOLDER = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//classification//"

label_to_id = {
    "hap":"0",
    "exc":"0",
    "sad":"1",
    "ang":"2",
    "neu":"3",
}

labels, features = load_dict_from_binary(TEST_EMBEDDINGS_LABELS, TEST_EMBEDDINGS, label_to_id)

experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, label_to_id, "classify_word_embeddings_dnn", use_timestamp=True)


label_dict, feature_dict = load_dict_from_binary(TRAIN_EMBEDDINGS_LABELS, TRAIN_EMBEDDINGS, label_to_id)
dnn.train(labels, features, False, experiment_dir, logger)


labels, features = load_from_binary(DEV_EMBEDDINGS_LABELS, DEV_EMBEDDINGS, label_to_id)
dnn.test(labels, features, False, experiment_dir, logger)

