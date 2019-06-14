from classification.util.experiments_util import *
from classification.util.dataset_utils import create_emobase_dataset_from_metadata, Normalization
from classification.audio import svm
from classification.util.global_vars import *

experiments_folder = os.path.join(ROOT_FOLDER, "experiments//audio")
dataset_path = os.path.join(ROOT_FOLDER, "datasets")

metadata = read_tsv_dataset( os.path.join(ROOT_FOLDER, "datasets//labels.tsv"))

class_groups = {
    "hap":0,
    "sad":1,
    "ang":2,
    "neu":3,
}

params = {
    "train_datasets": ['IEMOCAP']
}

experiment_dir, logger = create_experiment(experiments_folder, class_groups, "classify_svm_4_labels", use_timestamp=True)

train_dataset = create_emobase_dataset_from_metadata(metadata, class_groups, 'train', dataset_path, Normalization.CREATE_NORM, experiment_dir, params['train_datasets'])
dev_dataset = create_emobase_dataset_from_metadata(metadata, class_groups, 'dev', dataset_path, Normalization.USE_NORM, experiment_dir)

svm.train(train_dataset, dev_dataset, experiment_dir, logger)