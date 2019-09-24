from utils.experiments_util import *
from ms_dataset.ms_util import *
from classification.util.global_vars import *
from utils.rnn_utils import *
from models import LSTM
from utils.dataset_utils import create_sequence_dataset_from_metadata
import pickle

train_pickle = os.path.join(ROOT_FOLDER, 'datasets//MS//features//text//normalized_train_embedding_dataset.pkl')
dev_pickle = os.path.join(ROOT_FOLDER, 'datasets//MS//features//text//normalized_dev_embedding_dataset.pkl')
test_pickle = os.path.join(ROOT_FOLDER, 'datasets//MS//features//text//normalized_test_embedding_dataset.pkl')

train_metadata = read_ms_tsv_metadata(os.path.join(ROOT_FOLDER, 'datasets//MS//train.tsv'))
dev_metadata = read_ms_tsv_metadata(os.path.join(ROOT_FOLDER, 'datasets//MS//dev.tsv'))
test_metadata = read_ms_tsv_metadata(os.path.join(ROOT_FOLDER, 'datasets//MS//test.tsv'))

EXPERIMENTS_FOLDER = os.path.join(ROOT_FOLDER, 'experiments//ms//text')

class_groups = {
    "Neutral":0,
    "Positive":1,
    "Negative":2
}

params = {
    "batch_size": 64,
    "hidden_size": 512,
    "drop_prob": 0.0,
    "fully_connected_drop_prob": 0.0,
    "layers": 2,
    "learning_rate": 1e-5,
    "epochs": 1000
}

params["label_dim"] = len(set(list(class_groups.values())))
experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, class_groups, "MS_classify_embeddings", use_timestamp=True)

train_pkl = open(train_pickle, 'wb')
train_dataset = pickle.load(train_pkl)

dev_pkl = open(dev_pickle, 'wb')
dev_dataset = pickle.load(dev_pkl)

test_pkl = open(test_pickle, 'wb')
test_dataset = pickle.load(test_pkl)

params["input_dim"] = train_dataset.tensors[0][0].size()[1]

model = LSTM.LSTM(params)

id_to_name = {}
for m in train_metadata:
    id_to_name[int(m["Id"])] = m["Name"]
for m in dev_metadata:
    id_to_name[int(m["Id"])] = m["Name"]
for m in test_metadata:
    id_to_name[int(m["Id"])] = m["Name"]

train(train_dataset, dev_dataset, test_dataset, id_to_name, experiment_dir, model, logger, params)

