from utils.experiments_util import *
from classification.util.global_vars import *
from classification.text import naive_bayes

experiments_folder = os.path.join(ROOT_FOLDER, "IEMOCAP_txt//experiments")

train_txt = os.path.join(ROOT_FOLDER, "features//text//utterances//train.txt")
dev_txt = os.path.join(ROOT_FOLDER, "features//text//utterances//dev.txt")

label_to_id = {
    "hap":"0",
    "exc":"0",
    "sad":"1",
    "fru":"1",
    "ang":"1",
    "fea":"1",
    "dis":"1",
    "neu":"2",
}

experiment_dir, logger = create_experiment(experiments_folder, label_to_id, "classify_naive_bayes", use_timestamp=True)

naive_bayes.train(train_txt, experiment_dir, label_to_id, logger)
naive_bayes.test(dev_txt, experiment_dir, label_to_id, logger)