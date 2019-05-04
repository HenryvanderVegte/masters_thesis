from classification.util.experiments_util import *
from classification.util.global_vars import *
from classification.text.word_embeddings import *
import gensim
import os


experiments_folder = os.path.join(ROOT_FOLDER, "IEMOCAP_txt//experiments")


train_txt = os.path.join(ROOT_FOLDER, "features//text//utterances//train.txt")
dev_txt = os.path.join(ROOT_FOLDER, "features//text//utterances//dev.txt")
test_txt = os.path.join(ROOT_FOLDER, "features//text//utterances//test.txt")

model = gensim.models.KeyedVectors.load_word2vec_format('file://C://Users//Henry//Desktop//Masterarbeit//word_embeddings//GoogleNews-vectors-negative300.bin', binary=True)
oov_vector = np.random.normal(0, 0.1, model.vector_size)
label_to_id = {
    "hap":"0",
    "exc":"0",
    "sad":"1",
    "ang":"2",
    "neu":"3",
}

experiment_dir, logger = create_experiment(experiments_folder, label_to_id, "extract_word_embeddings_with_time", use_timestamp=True)

train_embeddings_path = os.path.join(experiment_dir, 'train_embeddings.npy')
train_embeddings_labels_path = os.path.join(experiment_dir, 'train_embeddings.txt')
extract_word_embeddings_with_time(train_txt, model, train_embeddings_labels_path, train_embeddings_path, oov_vector)

dev_embeddings_path = os.path.join(experiment_dir, 'dev_embeddings.npy')
dev_embeddings_labels_path = os.path.join(experiment_dir, 'dev_embeddings.txt')
extract_word_embeddings_with_time(dev_txt, model, dev_embeddings_labels_path, dev_embeddings_path, oov_vector)

test_embeddings_path = os.path.join(experiment_dir, 'test_embeddings.npy')
test_embeddings_labels_path = os.path.join(experiment_dir, 'test_embeddings.txt')
extract_word_embeddings_with_time(test_txt, model, test_embeddings_labels_path, test_embeddings_path, oov_vector)