from classification.util.experiments_util import *
from classification.audio import dnn
from classification.util.global_vars import *

experiments_folder = os.path.join(ROOT_FOLDER, "IEMOCAP_audio//experiments")

train_txt = os.path.join(ROOT_FOLDER, "features//audio//train.txt")
dev_txt = os.path.join(ROOT_FOLDER, "features//audio//dev.txt")

label_to_id = {
    "hap":"0",
    "sad":"1",
    "ang":"2",
    "neu":"3",
}

experiment_dir, logger = create_experiment(experiments_folder, label_to_id, "classify_dnn_4_labels", use_timestamp=True)

dnn.train(train_txt, experiment_dir, label_to_id, logger)

dnn.test(dev_txt, experiment_dir, label_to_id, logger)