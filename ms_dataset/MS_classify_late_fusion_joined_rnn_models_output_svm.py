from utils.experiments_util import *
from ms_dataset.ms_util import *
from global_vars import *
from utils.two_modality_utils import *
from models import LSTM
from utils.dataset_utils import create_sequence_dataset_from_metadata

params = {
    "batch_size": 64,
    "hidden_size": 256,
    "drop_prob": 0.0,
    "fully_connected_drop_prob": 0.0,
    "layers": 2,
    "learning_rate": 1e-2,
    "epochs": 1000
}

class_groups = {
    "Neutral":0,
    "Positive":1,
    "Negative":2
}

EXPERIMENTS_FOLDER = os.path.join(ROOT_FOLDER, 'experiments//ms//fusion')
params["label_dim"] = len(set(list(class_groups.values())))
experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, class_groups, "MS_classify_fusion_output_svm", use_timestamp=True)

train_metadata = read_ms_tsv_metadata(os.path.join(ROOT_FOLDER, 'datasets//MS//train.tsv'))
dev_metadata = read_ms_tsv_metadata(os.path.join(ROOT_FOLDER, 'datasets//MS//dev.tsv'))
test_metadata = read_ms_tsv_metadata(os.path.join(ROOT_FOLDER, 'datasets//MS//test.tsv'))


################################################## Load embeddings model: ##################################################
embeddings_model_path = os.path.join(ROOT_FOLDER, 'models//MS_classify_embeddings//model.pth')
embeddings_params = {
    "hidden_size": 256,
    "drop_prob": 0.0,
    "fully_connected_drop_prob": 0.0,
    "layers": 2,
}

ms_embeddings_train_path_normalized = os.path.join(ROOT_FOLDER, 'datasets//MS//features//text//embeddings_train_normalized.npy')
ms_embeddings_test_path_normalized = os.path.join(ROOT_FOLDER, 'datasets//MS//features//text//embeddings_test_normalized.npy')

embeddings_train_features = np.load(ms_embeddings_train_path_normalized).item()
embeddings_test_features = np.load(ms_embeddings_test_path_normalized).item()

embeddings_train_dataset = create_sequence_dataset_from_metadata(train_metadata, embeddings_train_features, class_groups)
embeddings_test_dataset = create_sequence_dataset_from_metadata(test_metadata, embeddings_test_features, class_groups)

embeddings_resources = {}
embeddings_resources['train_dataset'] = embeddings_train_dataset
embeddings_resources['test_dataset'] = embeddings_test_dataset

embeddings_params["input_dim"] = embeddings_resources['train_dataset'].tensors[0][0].size()[1]
embeddings_params["label_dim"] = len(set(list(class_groups.values())))

embeddings_model = LSTM.LSTM(embeddings_params)
embeddings_model.load_state_dict(torch.load(embeddings_model_path))
embeddings_model.eval()
embeddings_resources['model'] = embeddings_model

################################################## Load acoustic model: ##################################################
acoustic_model_path = os.path.join(ROOT_FOLDER, 'models//MS_classify_emobase//model.pth')
acoustic_params = {
    "hidden_size": 256,
    "drop_prob": 0.0,
    "fully_connected_drop_prob": 0.0,
    "layers": 2,
}

ms_acoustic_train_path_normalized = os.path.join(ROOT_FOLDER, 'datasets//MS//features//audio//emobase_word_level_test_normalized.npy')
ms_acoustic_test_path_normalized = os.path.join(ROOT_FOLDER, 'datasets//MS//features//audio//emobase_word_level_train_normalized.npy')

acoustic_train_features = np.load(ms_acoustic_train_path_normalized).item()
acoustic_test_features = np.load(ms_acoustic_test_path_normalized).item()

acoustic_train_dataset = create_sequence_dataset_from_metadata(train_metadata, acoustic_train_features, class_groups)
acoustic_test_dataset = create_sequence_dataset_from_metadata(test_metadata, acoustic_test_features, class_groups)

acoustic_resources = {}
acoustic_resources['train_dataset'] = acoustic_train_dataset
acoustic_resources['test_dataset'] = acoustic_test_dataset

acoustic_params["input_dim"] = acoustic_resources['train_dataset'].tensors[0][0].size()[1]
acoustic_params["label_dim"] = len(set(list(class_groups.values())))

acoustic_model = LSTM.LSTM(acoustic_params)
acoustic_model.load_state_dict(torch.load(acoustic_model_path))
acoustic_model.eval()
acoustic_resources['model'] = acoustic_model

id_to_name = {}
for m in train_metadata:
    id_to_name[int(m["Id"])] = m["Name"]
for m in dev_metadata:
    id_to_name[int(m["Id"])] = m["Name"]
for m in test_metadata:
    id_to_name[int(m["Id"])] = m["Name"]


train_two_modality_final_output_svm(embeddings_resources, acoustic_resources, id_to_name, experiment_dir, logger)