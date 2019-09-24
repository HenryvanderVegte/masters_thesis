from utils.experiments_util import *
from ms_dataset.ms_util import *
from classification.util.global_vars import *
from utils.rnn_utils import *
from models import LSTM
from utils.dataset_utils import create_sequence_dataset_from_metadata

ms_embeddings_train_path = os.path.join(ROOT_FOLDER, 'datasets//MS//features//text//embeddings_train.npy')
ms_embeddings_dev_path = os.path.join(ROOT_FOLDER, 'datasets//MS//features//text//embeddings_dev.npy')
ms_embeddings_test_path = os.path.join(ROOT_FOLDER, 'datasets//MS//features//text//embeddings_test.npy')

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
    "batch_size": 16,
    "hidden_size": 512,
    "drop_prob": 0.0,
    "fully_connected_drop_prob": 0.0,
    "layers": 2,
    "learning_rate": 1e-5,
    "epochs": 1000
}

params["label_dim"] = len(set(list(class_groups.values())))
experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, class_groups, "MS_classify_embeddings", use_timestamp=True)

embeddings_train_features = np.load(ms_embeddings_train_path).item()
embeddings_test_features = np.load(ms_embeddings_test_path).item()
embeddings_dev_features = np.load(ms_embeddings_dev_path).item()

means, stddevs = get_means_and_stddevs_from_sequence_dataset(train_metadata, embeddings_train_features, class_groups)

embeddings_train_features = normalize_dataset(embeddings_train_features, means, stddevs)
ms_embeddings_train_path_normalized = os.path.join(ROOT_FOLDER, 'datasets//MS//features//text//embeddings_train_normalized.npy')
np.save(ms_embeddings_train_path_normalized, embeddings_train_features)

embeddings_test_features = normalize_dataset(embeddings_test_features, means, stddevs)
ms_embeddings_test_path_normalized = os.path.join(ROOT_FOLDER, 'datasets//MS//features//audio//embeddings_test_normalized.npy')
np.save(ms_embeddings_test_path_normalized, embeddings_test_features)

embeddings_dev_features = normalize_dataset(embeddings_dev_features, means, stddevs)
ms_embeddings_dev_path_normalized = os.path.join(ROOT_FOLDER, 'datasets//MS//features//audio//embeddings_dev_normalized.npy')
np.save(ms_embeddings_dev_path_normalized, embeddings_dev_features)

train_dataset = create_sequence_dataset_from_metadata(train_metadata, embeddings_train_features, class_groups)
dev_dataset = create_sequence_dataset_from_metadata(dev_metadata, embeddings_dev_features, class_groups)
test_dataset = create_sequence_dataset_from_metadata(test_metadata, embeddings_test_features, class_groups)

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

