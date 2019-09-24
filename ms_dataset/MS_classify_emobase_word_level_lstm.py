from utils.experiments_util import *
from ms_dataset.ms_util import *
from classification.util.global_vars import *
from utils.rnn_utils import *
from models import LSTM
from utils.dataset_utils import create_sequence_dataset_from_metadata
import pickle

ms_emobase_train_path_normalized = os.path.join(ROOT_FOLDER, 'datasets//MS//features//audio//emobase_word_level_train_normalized.npy')
ms_emobase_dev_path_normalized = os.path.join(ROOT_FOLDER, 'datasets//MS//features//audio//emobase_word_level_dev_normalized.npy')
ms_emobase_test_path_normalized = os.path.join(ROOT_FOLDER, 'datasets//MS//features//audio//emobase_word_level_test_normalized.npy')

train_metadata = read_ms_tsv_metadata(os.path.join(ROOT_FOLDER, 'datasets//MS//train.tsv'))
dev_metadata = read_ms_tsv_metadata(os.path.join(ROOT_FOLDER, 'datasets//MS//dev.tsv'))
test_metadata = read_ms_tsv_metadata(os.path.join(ROOT_FOLDER, 'datasets//MS//test.tsv'))

EXPERIMENTS_FOLDER = os.path.join(ROOT_FOLDER, 'experiments//ms//audio')

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
    "learning_rate": 1e-3,
    "epochs": 1000
}

params["label_dim"] = len(set(list(class_groups.values())))
experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, class_groups, "MS_classify_emobase", use_timestamp=True)

emobase_train_features = np.load(ms_emobase_train_path_normalized).item()
emobase_dev_features = np.load(ms_emobase_dev_path_normalized).item()
emobase_test_features = np.load(ms_emobase_test_path_normalized).item()

train_dataset = create_sequence_dataset_from_metadata(train_metadata, emobase_train_features, class_groups)
train_pkl = os.path.join(ROOT_FOLDER, 'datasets//MS//features//audio//normalized_train_emobase_dataset.torch')
torch.save(train_dataset, train_pkl)


dev_dataset = create_sequence_dataset_from_metadata(dev_metadata, emobase_dev_features, class_groups)
dev_pkl = os.path.join(ROOT_FOLDER, 'datasets//MS//features//audio//normalized_dev_emobase_dataset.torch')
torch.save(dev_dataset, dev_pkl)

test_dataset = create_sequence_dataset_from_metadata(test_metadata, emobase_test_features, class_groups)
test_pkl = os.path.join(ROOT_FOLDER, 'datasets//MS//features//audio//normalized_test_emobase_dataset.torch')
torch.save(test_dataset, test_pkl)

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

