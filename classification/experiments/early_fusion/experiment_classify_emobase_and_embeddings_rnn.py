import classification.util.data_loader_pickle as data_loader
from classification.util.experiments_util import *
from classification.util.global_vars import *
from classification.util.rnn_utils import *
from classification.text import rnn_pretrained_embeddings
from classification.util.text_preprocessing import create_sequence_dataset_from_metadata

embeddings = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//audio//emobase_word_level_50ms_buffer.npy')
metadata = read_tsv_dataset(os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//labels.tsv'))
EXPERIMENTS_FOLDER = os.path.join(ROOT_FOLDER, 'experiments//fusion')

class_groups = {
    "hap":0,
    "exc":0,
    "sad":1,
    "ang":2,
    "neu":3,
}

params = {
    "batch_size": 16,
    "hidden_size": 32,
    "drop_prob": 0.1,
    "fully_connected_drop_prob": 0.4,
    "layers": 2,
    "epochs": 1000,
    "log_x_epochs": 2,
}

params["labels_size"] = len(set(list(class_groups.values())))

experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, class_groups, "classify_word_embeddings_normalization", use_timestamp=True)
embeddings = np.load(embeddings).item()
embeddings = normalize_features(embeddings)


train_dataset = create_sequence_dataset_from_metadata(metadata,embeddings, class_groups, 'train')
dev_dataset = create_sequence_dataset_from_metadata(metadata,embeddings, class_groups, 'dev')

params["embedding_dim"] = dev_dataset.tensors[0][0].size()[1]

model = rnn_pretrained_embeddings.PretrainedEmbeddingsLSTM(params)

id_to_name = {}
for m in metadata:
    id_to_name[int(m["Id"])] = m["Name"]

train(train_dataset, dev_dataset, id_to_name, experiment_dir, model, logger, params)
