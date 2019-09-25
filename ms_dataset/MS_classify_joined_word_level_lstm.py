from utils.experiments_util import *
from ms_dataset.ms_util import *
from global_vars import *
from utils.rnn_utils import *
from models import LSTM
from utils.dataset_utils import create_sequence_dataset_from_metadata

train_metadata = read_ms_tsv_metadata(os.path.join(ROOT_FOLDER, 'datasets//MS//train.tsv'))
dev_metadata = read_ms_tsv_metadata(os.path.join(ROOT_FOLDER, 'datasets//MS//dev.tsv'))
test_metadata = read_ms_tsv_metadata(os.path.join(ROOT_FOLDER, 'datasets//MS//test.tsv'))

ms_embeddings_train_path_normalized = os.path.join(ROOT_FOLDER, 'datasets//MS//features//text//embeddings_train_normalized.npy')
ms_embeddings_dev_path_normalized = os.path.join(ROOT_FOLDER, 'datasets//MS//features//text//embeddings_dev_normalized.npy')
ms_embeddings_test_path_normalized = os.path.join(ROOT_FOLDER, 'datasets//MS//features//text//embeddings_test_normalized.npy')

ms_emobase_train_path_normalized = os.path.join(ROOT_FOLDER, 'datasets//MS//features//audio//emobase_word_level_train_normalized.npy')
ms_emobase_dev_path_normalized = os.path.join(ROOT_FOLDER, 'datasets//MS//features//audio//emobase_word_level_dev_normalized.npy')
ms_emobase_test_path_normalized = os.path.join(ROOT_FOLDER, 'datasets//MS//features//audio//emobase_word_level_test_normalized.npy')

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
    "learning_rate": 1e-3,
    "epochs": 1000
}

params["label_dim"] = len(set(list(class_groups.values())))
experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, class_groups, "MS_classify_emobase", use_timestamp=True)

embeddings_train_features = np.load(ms_embeddings_train_path_normalized).item()
embeddings_dev_features = np.load(ms_embeddings_dev_path_normalized).item()
embeddings_test_features = np.load(ms_embeddings_test_path_normalized).item()

emobase_train_features = np.load(ms_emobase_train_path_normalized).item()
emobase_dev_features = np.load(ms_emobase_dev_path_normalized).item()
emobase_test_features = np.load(ms_emobase_test_path_normalized).item()

joined_train_features = join_feature_dicts(embeddings_train_features, emobase_train_features)
joined_dev_features = join_feature_dicts(embeddings_dev_features, emobase_dev_features)
joined_test_features = join_feature_dicts(embeddings_test_features, emobase_test_features)

train_dataset = create_sequence_dataset_from_metadata(train_metadata, joined_train_features, class_groups)
dev_dataset = create_sequence_dataset_from_metadata(dev_metadata, joined_dev_features, class_groups)
test_dataset = create_sequence_dataset_from_metadata(test_metadata, joined_test_features, class_groups)

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
