from classification.util.experiments_util import *
from classification.text.word_embeddings import *
import gensim
import os



EXPERIMENTS_FOLDER = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//classification//"
TRAIN_TXT = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//split//train.txt"
DEV_TXT = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//split//dev.txt"
TEST_TXT = "C://Users//Henry//Desktop//Masterarbeit//IEMOCAP_txt//split//test.txt"

model = gensim.models.KeyedVectors.load_word2vec_format('file://C://Users//Henry//Desktop//Masterarbeit//word_embeddings//GoogleNews-vectors-negative300.bin', binary=True)
oov_vector = np.random.normal(0, 0.1, model.vector_size)
label_to_id = {
    "hap":"0",
    "exc":"0",
    "sad":"1",
    "ang":"2",
    "neu":"3",
}

experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, label_to_id, "extract_word_embeddings_with_time", use_timestamp=True)

train_embeddings_path = os.path.join(experiment_dir, 'train_embeddings.npy')
train_embeddings_labels_path = os.path.join(experiment_dir, 'train_embeddings.txt')
extract_word_embeddings_with_time(TRAIN_TXT, model, train_embeddings_labels_path, train_embeddings_path, oov_vector)

dev_embeddings_path = os.path.join(experiment_dir, 'dev_embeddings.npy')
dev_embeddings_labels_path = os.path.join(experiment_dir, 'dev_embeddings.txt')
extract_word_embeddings_with_time(DEV_TXT, model, dev_embeddings_labels_path, dev_embeddings_path, oov_vector)

test_embeddings_path = os.path.join(experiment_dir, 'test_embeddings.npy')
test_embeddings_labels_path = os.path.join(experiment_dir, 'test_embeddings.txt')
extract_word_embeddings_with_time(TEST_TXT, model, test_embeddings_labels_path, test_embeddings_path, oov_vector)