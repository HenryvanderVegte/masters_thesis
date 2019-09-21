from utils.experiments_util import *
from utils.naive_bayes_utils import *
from classification.util.global_vars import *

metadata = read_tsv_metadata(os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//labels.tsv'))
EXPERIMENTS_FOLDER = os.path.join(ROOT_FOLDER, 'experiments//text')

class_groups = {
    "hap":0,
    "exc":0,
    "sad":1,
    "ang":2,
    "neu":3,
}

experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, class_groups, "CV_classify_naive_bayes", use_timestamp=True)

nr_of_folds = 10

all_golds = []
all_preds = []
for i in range(0, nr_of_folds):
    test_fold_nr = i
    validation_fold_nr = (i + 1) % nr_of_folds

    test_folds = [test_fold_nr]
    train_folds = list(range(0, nr_of_folds))
    train_folds.remove(test_fold_nr)
    train_folds.remove(validation_fold_nr)

    logger.info('Testing on folds: ' + str(test_folds))
    logger.info('Training on folds: ' + str(train_folds))

    all_words_set = create_top_n_words_set_from_metadata(metadata, class_groups, train_folds, top_n=100)
    train_dataset = create_dataset_from_metadata(metadata, class_groups, all_words_set, train_folds)
    test_dataset = create_dataset_from_metadata(metadata, class_groups, all_words_set, test_folds)

    fold_path = os.path.join(experiment_dir, str(i))
    os.mkdir(fold_path)
    test_golds, test_preds = train(train_dataset, test_dataset, fold_path, logger)
    all_golds += test_golds
    all_preds += test_preds

logger.info('!Final result!:')
metrics_str = get_metrics_str(all_golds, all_preds)
logger.info(metrics_str)