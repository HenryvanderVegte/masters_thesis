from utils.experiments_util import *
from global_vars import *
from utils.mixed_modelkind_utils import *
from models import LSTM
from utils.dataset_utils import create_sequence_dataset_from_metadata, create_dataset_from_metadata

embedding_features_path = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//text//google_news_word_embeddings.npy')
emobase_features_path = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//audio//emobase_utterance_level.npy')
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

emobase_params = {}
emobase_params["label_dim"] = len(set(list(class_groups.values())))

experiment_params = {
    "batch_size": 32,
}

nr_of_folds = 10

all_golds = []
all_preds = []
for i in range(0, nr_of_folds):
    logger.info('Testing on fold ' + str(i))
    test_fold = [i]

    train_folds = list(range(0, nr_of_folds))
    train_folds.remove(i)

    embedding_features = np.load(embedding_features_path).item()
    means, stddevs = get_means_and_stddevs_from_sequence_dataset(metadata, embedding_features, class_groups, train_folds)
    embedding_features = normalize_dataset(embedding_features, means, stddevs)

    word_embedding_resources = {}
    word_embedding_resources['train_dataset'] = create_sequence_dataset_from_metadata(metadata, embedding_features, class_groups, train_folds)
    word_embedding_resources['test_dataset']  = create_sequence_dataset_from_metadata(metadata, embedding_features, class_groups, test_fold)
    word_embedding_params["input_dim"] = word_embedding_resources['train_dataset'].tensors[0][0].size()[1]
    word_embedding_params["label_dim"] = len(set(list(class_groups.values())))
    word_embedding_model_path = os.path.join(ROOT_FOLDER, 'models//CV//4//CV_classify_word_embeddings////' + str(i) + '//model.pth')
    logger.info(word_embedding_model_path)
    word_embedding_model = LSTM.LSTM(word_embedding_params)
    word_embedding_model.load_state_dict(torch.load(word_embedding_model_path))
    word_embedding_model.eval()
    word_embedding_resources['model'] = word_embedding_model

    emobase_features = np.load(emobase_features_path).item()
    means, stddevs = get_means_and_stddevs_from_dataset(metadata, embedding_features, class_groups, train_folds)
    emobase_features = normalize_dataset(emobase_features, means, stddevs)

    emobase_resources = {}
    emobase_resources['train_dataset'] = create_dataset_from_metadata(metadata, emobase_features, class_groups, train_folds)
    emobase_resources['test_dataset'] = create_dataset_from_metadata(metadata, emobase_features, class_groups, test_fold)

    emobase_model_path = os.path.join(ROOT_FOLDER, 'models//others//CV_classify_emobase_utterance_level_svm//' + str(i) + '//svm_model.pkl')
    logger.info(emobase_model_path)
    emobase_model = pickle.load(open(emobase_model_path, 'rb'))
    emobase_resources['model'] = emobase_model

    id_to_name = {}
    for m in metadata:
        id_to_name[int(m["Id"])] = m["Name"]

    fold_path = os.path.join(experiment_dir, str(i))
    os.mkdir(fold_path)
    test_golds, test_preds = train_mixed_modality_final_output_svm(word_embedding_resources, emobase_resources, id_to_name, fold_path, logger, experiment_params)
    all_golds += test_golds
    all_preds += test_preds

logger.info('!Final result!:')
metrics_str = get_metrics_str(all_golds, all_preds)
logger.info(metrics_str)
