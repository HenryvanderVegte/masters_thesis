from utils.experiments_util import *
from classification.util.global_vars import *
from utils.rnn_utils import *
from models import LSTM
from utils.dataset_utils import create_sequence_dataset_from_metadata

emobase_features = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//audio//emobase_word_level_50ms_buffer.npy')
metadata = read_tsv_metadata(os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//labels.tsv'))
EXPERIMENTS_FOLDER = os.path.join(ROOT_FOLDER, 'experiments//audio')

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
    "fully_connected_drop_prob": 0.2,
    "layers": 2,
    "epochs": 1000
}

params["label_dim"] = len(set(list(class_groups.values())))
experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, class_groups, "classify_emobase_50ms_buffer", use_timestamp=True)

emobase_features = np.load(emobase_features).item()
emobase_features = normalize_sequence_features(emobase_features)

train_dataset = create_sequence_dataset_from_metadata(metadata, emobase_features, class_groups, [0, 1, 2, 3])
validation_dataset = create_sequence_dataset_from_metadata(metadata, emobase_features, class_groups, [4])
test_dataset = create_sequence_dataset_from_metadata(metadata, emobase_features, class_groups, [5])

params["input_dim"] = train_dataset.tensors[0][0].size()[1]

model = LSTM.LSTM(params)

id_to_name = {}
for m in metadata:
    id_to_name[int(m["Id"])] = m["Name"]

train(train_dataset, validation_dataset, test_dataset, id_to_name, experiment_dir, model, logger, params)