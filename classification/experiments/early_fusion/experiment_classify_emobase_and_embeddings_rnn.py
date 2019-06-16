import classification.util.data_loader_pickle as data_loader
from classification.util.experiments_util import *
from classification.util.global_vars import *
from classification.util.rnn_utils import *
from classification.text import rnn_pretrained_embeddings
from classification.util.text_preprocessing import create_sequence_dataset_from_metadata

embeddings = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//fusion//emobase_and_embeddings.npy')
metadata = read_tsv_dataset(os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//labels.tsv'))
EXPERIMENTS_FOLDER = "C://Users//Henry//Desktop//Masterarbeit//experiments//fusion//"

class_groups = {
    "hap":0,
    "exc":0,
    "sad":1,
    "ang":2,
    "neu":3,
}

params = {
    "max_sequence_length": 50,
    "batch_size": 16,
    "hidden_size": 128,
    "drop_prob": 0.1,
    "fully_connected_drop_prob": 0.5,
    "layers": 2,
    "epochs": 1000,
    "log_x_epochs": 10,
}

params["labels_size"] = len(set(list(class_groups.values())))

experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, class_groups, "classify_word_embeddings_rnn", use_timestamp=True)
embeddings = np.load(embeddings).item()

train_dataset = create_sequence_dataset_from_metadata(metadata,embeddings, class_groups, 'train')
dev_dataset = create_sequence_dataset_from_metadata(metadata,embeddings, class_groups, 'dev')

params["embedding_dim"] = dev_dataset.tensors[0][0].size()[1]

model = rnn_pretrained_embeddings.PretrainedEmbeddingsLSTM(params)

id_to_name = {}
for m in metadata:
    id_to_name[int(m["Id"])] = m["Name"]

train(train_dataset, dev_dataset, id_to_name, experiment_dir, model, logger, params)
