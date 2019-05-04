from classification.util.data_loader_pickle import *
from classification.util.experiments_util import *
from classification.audio import svm

TRAIN_EMBEDDINGS_LABELS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//embeddings//train_embeddings.txt"
TRAIN_EMBEDDINGS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//embeddings//train_embeddings.npy"

DEV_EMBEDDINGS_LABELS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//embeddings//dev_embeddings.txt"
DEV_EMBEDDINGS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//embeddings//dev_embeddings.npy"

TEST_EMBEDDINGS_LABELS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//embeddings//test_embeddings.txt"
TEST_EMBEDDINGS = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//embeddings//test_embeddings.npy"

EXPERIMENTS_FOLDER = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//classification//"

label_to_id = {
    "hap":"0",
    "exc":"0",
    "sad":"1",
    "ang":"2",
    "neu":"3",
}

experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, label_to_id, "classify_word_embeddings_svm", use_timestamp=True)


labels, features = load_from_binary(TRAIN_EMBEDDINGS_LABELS, TRAIN_EMBEDDINGS, label_to_id)
svm.train(labels, features, False, experiment_dir, logger)


labels, features = load_from_binary(DEV_EMBEDDINGS_LABELS, DEV_EMBEDDINGS, label_to_id)
svm.test(labels, features, False, experiment_dir, logger)

