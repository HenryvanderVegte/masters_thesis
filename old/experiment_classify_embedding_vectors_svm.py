from old.data_loader_pickle import *
from utils.experiments_util import *
from global_vars import *
from classification.audio import svm

experiments_folder = os.path.join(ROOT_FOLDER, "IEMOCAP_txt//experiments")

train_embedding_labels = os.path.join(ROOT_FOLDER, "features//text//embeddings//1-dim//train_embeddings.txt")
train_embeddings = os.path.join(ROOT_FOLDER, "features//text//embeddings//1-dim//train_embeddings.npy")

dev_embedding_labels = os.path.join(ROOT_FOLDER, "features//text//embeddings//1-dim//dev_embeddings.txt")
dev_embeddings = os.path.join(ROOT_FOLDER, "features//text//embeddings//1-dim//dev_embeddings.npy")

label_to_id = {
    "hap":"0",
    "exc":"0",
    "sad":"1",
    "ang":"2",
    "neu":"3",
}

experiment_dir, logger = create_experiment(experiments_folder, label_to_id, "classify_word_embeddings_svm", use_timestamp=True)


labels, features = load_from_binary(train_embedding_labels, train_embeddings, label_to_id)
svm.train(labels, features, False, experiment_dir, logger)


labels, features = load_from_binary(dev_embedding_labels, dev_embeddings, label_to_id)
svm.test(labels, features, False, experiment_dir, logger)

