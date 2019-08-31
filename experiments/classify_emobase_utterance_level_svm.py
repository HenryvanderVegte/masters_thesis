from utils.experiments_util import *
from classification.util.global_vars import *
from utils.svm_utils import *
from utils.dataset_utils import create_dataset_from_metadata

emobase_features = os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//features//audio//emobase_utterance_level.npy')
metadata = read_tsv_metadata(os.path.join(ROOT_FOLDER, 'datasets//IEMOCAP//labels.tsv'))
EXPERIMENTS_FOLDER = os.path.join(ROOT_FOLDER, 'experiments//audio')

class_groups = {
    "hap":0,
    "exc":0,
    "sad":1,
    "ang":2,
    "neu":3,
}

experiment_dir, logger = create_experiment(EXPERIMENTS_FOLDER, class_groups, "classify_emobase_utterance_level_svm", use_timestamp=True)
emobase_features = np.load(emobase_features).item()
emobase_features = normalize_features(emobase_features)

dev_dataset = create_dataset_from_metadata(metadata, emobase_features, class_groups, 'dev')
train_dataset = create_dataset_from_metadata(metadata, emobase_features, class_groups, 'train')

train(train_dataset, dev_dataset, experiment_dir, logger)