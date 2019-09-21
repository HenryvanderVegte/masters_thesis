from utils.experiments_util import *
from classification.util.global_vars import *
from utils.svm_utils import *
from utils.dataset_utils import create_dataset_from_metadata

emobase_features_path = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//audio//emobase_utterance_level.npy')
metadata = read_tsv_metadata(os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//labels.tsv'))
EXPERIMENTS_FOLDER = os.path.join(ROOT_FOLDER, 'experiments//audio')

class_groups = {
    "hap":0,
    "exc":0,
    "sad":1,
    "ang":2,
    "neu":3,
}

experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, class_groups, "CV_classify_emobase_utterance_level_svm", use_timestamp=True)

nr_of_folds = 10

id_to_name = {}
for m in metadata:
    id_to_name[int(m["Id"])] = m["Name"]

all_golds = []
all_preds = []
for i in range(0, nr_of_folds):
    test_fold_nr = i
    validation_fold_nr = (i + 1) % nr_of_folds

    logger.info('Testing on fold ' + str(test_fold_nr))
    test_fold = [test_fold_nr]

    train_folds = list(range(0, nr_of_folds))
    train_folds.remove(test_fold_nr)
    train_folds.remove(validation_fold_nr)

    emobase_features = np.load(emobase_features_path).item()
    means, stddevs = get_means_and_stddevs_from_dataset(metadata, emobase_features, class_groups, train_folds)
    emobase_features = normalize_dataset(emobase_features, means, stddevs)

    train_dataset = create_dataset_from_metadata(metadata, emobase_features, class_groups, train_folds)
    test_dataset = create_dataset_from_metadata(metadata, emobase_features, class_groups, test_fold)

    fold_path = os.path.join(experiment_dir, str(i))
    os.mkdir(fold_path)

    test_golds, test_preds = train(train_dataset, test_dataset, id_to_name, fold_path, logger)
    all_golds += test_golds
    all_preds += test_preds

logger.info('!Final result!:')
metrics_str = get_metrics_str(all_golds, all_preds)
logger.info(metrics_str)
