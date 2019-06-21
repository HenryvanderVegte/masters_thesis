from classification.util.experiments_util import *
from classification.util.dataset_utils import create_emobase_dataset_from_metadata, Normalization
from classification.util.dnn_utils import *
from classification.audio import dnn
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
    "batch_size": 32,
    "epochs": 100,
    "log_x_epochs": 1,
    "train_datasets": ['SAVEE','RAVDESS']
}

experiment_dir, logger = create_experiment(experiments_folder, class_groups, "classify_dnn_4_labels", use_timestamp=True)

train_dataset = create_emobase_dataset_from_metadata(metadata, class_groups, 'train', dataset_path, Normalization.CREATE_NORM, experiment_dir, params['train_datasets'])
dev_dataset = create_emobase_dataset_from_metadata(metadata, class_groups, 'dev', dataset_path, Normalization.USE_NORM, experiment_dir)
input_size = dev_dataset.tensors[0].size()[1]

model = dnn.DNN(input_size, 4)

train(train_dataset, dev_dataset, experiment_dir, model, logger, params)
