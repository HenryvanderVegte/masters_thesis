from utils.experiments_util import *
from classification.util.global_vars import *
from utils.two_modality_utils import *
from models import LSTM
from utils.dataset_utils import create_sequence_dataset_from_metadata

metadata = read_tsv_metadata(os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//labels.tsv'))
EXPERIMENTS_FOLDER = os.path.join(ROOT_FOLDER, 'experiments//fusion')

class_groups = {
    "hap":0,
    "exc":0,
    "sad":1,
    "ang":2,
    "neu":3,
}
experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, class_groups, "CV_late_fusion_joined_model_output_svm", use_timestamp=True)

word_embedding_params = {
    "hidden_size": 32,
    "drop_prob": 0.0,
    "fully_connected_drop_prob": 0.2,
    "layers": 2,
}
word_embedding_params["label_dim"] = len(set(list(class_groups.values())))

word_embeddings_dataset_path = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//text//google_news_word_embeddings.npy')
word_embeddings_dataset = np.load(word_embeddings_dataset_path).item()
word_embeddings_dataset = normalize_sequence_features(word_embeddings_dataset, class_groups, metadata)

emobase_params = {
    "hidden_size": 32,
    "drop_prob": 0.0,
    "fully_connected_drop_prob": 0.2,
    "layers": 2,
}
emobase_params["label_dim"] = len(set(list(class_groups.values())))
emobase_dataset_path = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//audio//emobase_word_level_50ms_buffer.npy')
emobase_dataset = np.load(emobase_dataset_path).item()
emobase_dataset = normalize_sequence_features(emobase_dataset, class_groups, metadata)

nr_of_folds = 10

all_golds = []
all_preds = []

for i in range(0, nr_of_folds):
    test_fold_nr = i
    #validation fold not used for svm
    validation_fold_nr = (i + 1) % nr_of_folds

    train_folds = list(range(0, nr_of_folds))
    train_folds.remove(i)
    train_folds.remove(validation_fold_nr)

    test_folds = [test_fold_nr]

    logger.info('Testing on folds: ' + str(test_folds))
    logger.info('Training on folds: ' + str(train_folds))

    word_embedding_resources = {}
    word_embedding_resources['train_dataset'] = create_sequence_dataset_from_metadata(metadata, word_embeddings_dataset, class_groups, train_folds)
    word_embedding_resources['test_dataset']  = create_sequence_dataset_from_metadata(metadata, word_embeddings_dataset, class_groups, test_folds)
    word_embedding_params["input_dim"] = word_embedding_resources['train_dataset'].tensors[0][0].size()[1]
    word_embedding_params["label_dim"] = len(set(list(class_groups.values())))
    word_embedding_model_path = os.path.join(ROOT_FOLDER, 'models//CV//9//CV_classify_word_embeddings////' + str(i) + '//model.pth')
    logger.info(word_embedding_model_path)
    word_embedding_model = LSTM.LSTM(word_embedding_params)
    word_embedding_model.load_state_dict(torch.load(word_embedding_model_path))
    word_embedding_model.eval()
    word_embedding_resources['model'] = word_embedding_model

    emobase_resources = {}
    emobase_resources['train_dataset'] = create_sequence_dataset_from_metadata(metadata, emobase_dataset, class_groups, train_folds)
    emobase_resources['test_dataset'] = create_sequence_dataset_from_metadata(metadata, emobase_dataset, class_groups, test_folds)

    emobase_model_path = os.path.join(ROOT_FOLDER, 'models//CV//9//CV_classify_emobase_50ms_buffer//' + str(i) + '//model.pth')
    logger.info(emobase_model_path)
    emobase_params["input_dim"] = emobase_resources['train_dataset'].tensors[0][0].size()[1]
    emobase_params["label_dim"] = len(set(list(class_groups.values())))
    emobase_model = LSTM.LSTM(emobase_params)
    emobase_model.load_state_dict(torch.load(emobase_model_path))
    emobase_model.eval()
    emobase_resources['model'] = emobase_model

    id_to_name = {}
    for m in metadata:
        id_to_name[int(m["Id"])] = m["Name"]

    fold_path = os.path.join(experiment_dir, str(i))
    os.mkdir(fold_path)
    test_golds, test_preds = train_two_modality_final_output_svm(word_embedding_resources, emobase_resources, id_to_name, fold_path, logger)
    all_golds += test_golds
    all_preds += test_preds

logger.info('!Final result!:')
metrics_str = get_metrics_str(all_golds, all_preds)
logger.info(metrics_str)
