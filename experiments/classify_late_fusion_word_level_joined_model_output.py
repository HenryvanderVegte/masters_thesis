from utils.experiments_util import *
from classification.util.global_vars import *
from models import LSTM
from utils.dataset_utils import create_sequence_dataset_from_metadata
from utils.two_modality_utils import *

'''
This experiment takes two previously trained model (one acoustic and one textual) and creates a support vector machine
based on the final output of the two models.
'''

metadata = read_tsv_metadata(os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//labels.tsv'))
id_to_name = {}
for m in metadata:
    id_to_name[int(m["Id"])] = m["Name"]
EXPERIMENTS_FOLDER = os.path.join(ROOT_FOLDER, 'experiments//fusion')

class_groups = {
    "hap":0,
    "exc":0,
    "sad":1,
    "ang":2,
    "neu":3,
}

experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, class_groups, "classify_joined_word_level_model", use_timestamp=True)

# Set up pretrained text model (word embeddings)
word_embeddings_dataset_path = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//text//google_news_word_embeddings.npy')
word_embeddings_dataset = np.load(word_embeddings_dataset_path).item()
word_embeddings_dataset = normalize_sequence_features(word_embeddings_dataset)
word_embedding_dataset_train = create_sequence_dataset_from_metadata(metadata,word_embeddings_dataset, class_groups, 'train')
word_embedding_dataset_test = create_sequence_dataset_from_metadata(metadata,word_embeddings_dataset, class_groups, 'dev')

word_embedding_params = {
    "hidden_size": 32,
    "drop_prob": 0.1,
    "fully_connected_drop_prob": 0.2,
    "layers": 2,
}

word_embedding_params["input_dim"] = word_embedding_dataset_train.tensors[0][0].size()[1]
word_embedding_params["label_dim"] = len(set(list(class_groups.values())))
embedding_model_path = os.path.join(ROOT_FOLDER, 'models//word_embedding_models//0//rnn.pth')

word_embedding_model = LSTM.LSTM(word_embedding_params)
word_embedding_model.load_state_dict(torch.load(embedding_model_path))
word_embedding_model.eval()

word_embedding_resources = {}
word_embedding_resources['model'] = word_embedding_model
word_embedding_resources['train_dataset'] = word_embedding_dataset_train
word_embedding_resources['dev_dataset'] = word_embedding_dataset_test

# Set up pretrained audio model (emobase features on word level)
emobase_dataset_path = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//audio//emobase_word_level_50ms_buffer.npy')
emobase_dataset = np.load(emobase_dataset_path).item()
emobase_dataset = normalize_sequence_features(emobase_dataset)
emobase_dataset_train = create_sequence_dataset_from_metadata(metadata,emobase_dataset, class_groups, 'train')
emobase_dataset_test = create_sequence_dataset_from_metadata(metadata,emobase_dataset, class_groups, 'dev')

emobase_params = {
    "hidden_size": 32,
    "drop_prob": 0.1,
    "fully_connected_drop_prob": 0.2,
    "layers": 2,
}

emobase_params["input_dim"] = emobase_dataset_train.tensors[0][0].size()[1]
emobase_params["label_dim"] = len(set(list(class_groups.values())))
emobase_model_path = os.path.join(ROOT_FOLDER, 'models//emobase_models//0//rnn.pth')

emobase_model = LSTM.LSTM(emobase_params)
emobase_model.load_state_dict(torch.load(emobase_model_path))
emobase_model.eval()

emobase_resources = {}
emobase_resources['model'] = emobase_model
emobase_resources['train_dataset'] = emobase_dataset_train
emobase_resources['dev_dataset'] = emobase_dataset_test

params = {
    "batch_size": 16,
}

train_two_modality_final_output_svm(word_embedding_resources, emobase_resources, experiment_dir, logger, params)