from classification.util.experiments_util import *
from classification.util.global_vars import *
from utils.dnn_utils import *
from models import DNN
from classification.util.dataset_utils import create_dataset_from_metadata

emobase_features = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//audio//emobase_utterance_level.npy')
metadata = read_tsv_dataset(os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//labels.tsv'))
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
    "drop_prob": 0.4,
    "epochs": 1000,
    "log_x_epochs": 2,
}

params["label_dim"] = len(set(list(class_groups.values())))

experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, class_groups, "classify_emobase_utterance_level_dnn", use_timestamp=True)
emobase_features = np.load(emobase_features).item()
emobase_features = normalize_features(emobase_features)

dev_dataset = create_dataset_from_metadata(metadata, emobase_features, class_groups, 'dev')
train_dataset = create_dataset_from_metadata(metadata, emobase_features, class_groups, 'train')

params["input_dim"] = dev_dataset.tensors[0].size()[1]

model = DNN.DNN(params)

id_to_name = {}
for m in metadata:
    id_to_name[int(m["Id"])] = m["Name"]

train(train_dataset, dev_dataset, experiment_dir, model, logger, params)
