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
    "hidden_size": 64,
    "drop_prob": 0.1,
    "fully_connected_drop_prob": 0.2,
    "layers": 2,
    "epochs": 1000
}

params["label_dim"] = len(set(list(class_groups.values())))
experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, class_groups, "CV_classify_emobase_50ms_buffer", use_timestamp=True)

emobase_features = np.load(emobase_features).item()
emobase_features = normalize_sequence_features(emobase_features)

nr_of_folds = 10

all_golds = []
all_preds = []

for i in range(1, nr_of_folds):
    test_fold_nr = i
    validation_fold_nr = (i + 1) % nr_of_folds

    logger.info('Testing on fold ' + str(test_fold_nr))
    logger.info('Validating on fold ' + str(validation_fold_nr))

    validation_fold = [validation_fold_nr]
    test_fold = [test_fold_nr]

    train_folds = list(range(0, nr_of_folds))
    train_folds.remove(i)
    train_folds.remove(validation_fold_nr)

    train_dataset = create_sequence_dataset_from_metadata(metadata, emobase_features, class_groups, train_folds)
    validation_dataset = create_sequence_dataset_from_metadata(metadata, emobase_features, class_groups, validation_fold)
    test_dataset = create_sequence_dataset_from_metadata(metadata, emobase_features, class_groups, test_fold)

    params["input_dim"] = train_dataset.tensors[0][0].size()[1]

    model = LSTM.LSTM(params)

    id_to_name = {}
    for m in metadata:
        id_to_name[int(m["Id"])] = m["Name"]

    fold_path = os.path.join(experiment_dir, str(i))
    os.mkdir(fold_path)
    test_golds, test_preds = train(train_dataset, validation_dataset, test_dataset, id_to_name, fold_path, model, logger, params)
    all_golds += test_golds
    all_preds += test_preds

logger.info('!Final result!:')
metrics_str = get_metrics_str(all_golds, all_preds)
logger.info(metrics_str)
